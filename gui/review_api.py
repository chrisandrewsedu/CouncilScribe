"""Load an already-processed meeting into a ReviewPageData, and (Slice 2b)
write review edits back to disk.

Mirrors run_local's --review loading: Meeting.from_dict(transcript_named.json),
embeddings.json, load_profiles(), then review.build_review_state(). Write-back
(persist_review / apply_rename) mirrors run_local's --review save + _apply_gate:
mutations go through src.review, then transcript_named.json is written
(authoritative) with best-effort re-export + gate recompute."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src import config
from src.event_kinds import local_roles_for
# The crash-safe writer lives in src/ so run_local.py and src/* can use it too;
# re-exported under the old private name for this module's existing callers.
from src.atomic_io import atomic_write_text as _atomic_write_text
from src.models import Meeting

from gui.models import CONFIDENT_THRESHOLD, ENROLL_MIN_SPEECH_SECONDS, ReviewPageData, SpeakerCard
from gui.paths import is_safe_meeting_id

# Video container preference order (same set run_local.find_video_file checks).
_VIDEO_EXTS = (".m4v", ".mp4", ".mkv", ".webm", ".avi", ".mov")
# Merge-control marks per voice verdict: same voice / ambiguous / different people.
_VERDICT_MARK = {"match": "✓", "uncertain": "~", "mismatch": "✗", "unknown": "?"}
_LEAD_IN = 3.0  # seconds of context before a clip, mirroring run_local._review_seek


def find_meeting_media(meeting_dir: Path) -> Optional[tuple[str, str]]:
    """(kind, filename) for the best playable media: video if present, else the
    compressed audio.opus (left after cleanup), else audio.wav, else None.
    kind is 'video' or 'audio'."""
    for ext in _VIDEO_EXTS:
        candidate = meeting_dir / f"source{ext}"
        if candidate.exists():
            return "video", candidate.name
    if (meeting_dir / "audio.opus").exists():
        return "audio", "audio.opus"
    if (meeting_dir / "audio.wav").exists():
        return "audio", "audio.wav"
    return None


def _seek(candidate: float, *, is_video: bool, clip_offset: float) -> float:
    """Seek position in the SERVED media. audio.wav is clip-local; the source
    video is the full recording, so clip-local candidates need clip_offset added."""
    base = max(0.0, candidate - _LEAD_IN)
    return base + (clip_offset if is_video else 0.0)


def _load_roster_for(meeting_dir: Path):
    """Load the meeting's roster (by persisted body_slug) for name normalization,
    or None. Best-effort — never raises."""
    state_file = meeting_dir / "pipeline_state.json"
    body_slug = None
    if state_file.exists():
        try:
            body_slug = json.loads(state_file.read_text(encoding="utf-8")).get("body_slug")
        except (ValueError, OSError, AttributeError):
            body_slug = None
    if not body_slug:
        return None
    try:
        from src.roster import load_roster
        return load_roster(body_slug=body_slug)
    except Exception:
        return None


def _load_meeting_ctx(meeting_id: str):
    """(meeting, meeting_dir, roster) for a write-back, or None if unsafe/missing/malformed."""
    if not is_safe_meeting_id(meeting_id):
        return None
    meeting_dir = config.MEETINGS_DIR / meeting_id
    named = meeting_dir / "transcript_named.json"
    if not named.exists():
        return None
    try:
        meeting = Meeting.from_dict(json.loads(named.read_text(encoding="utf-8")))
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return None
    return meeting, meeting_dir, _load_roster_for(meeting_dir)


def _load_embeddings(meeting_dir: Path) -> dict:
    """embeddings.json -> {label: np.ndarray}, or {} if absent/malformed."""
    emb_path = meeting_dir / "embeddings.json"
    if not emb_path.exists():
        return {}
    try:
        return {k: np.array(v) for k, v in json.loads(emb_path.read_text()).items()}
    except (ValueError, OSError, TypeError, AttributeError):
        return {}


def persist_review(meeting, meeting_dir: Path, embeddings: dict | None = None) -> None:
    """Persist review edits. Always: sync segments + write transcript_named.json
    (authoritative). When embeddings is given (a merge relabeled segments +
    combined embeddings), also rewrite diarization.json + embeddings.json,
    mirroring run_local._persist_after_review. Export + gate are best-effort."""
    for seg in meeting.segments:
        m = meeting.speakers.get(seg.speaker_label)
        if m and m.speaker_name:
            seg.speaker_name = m.speaker_name
            seg.confidence = m.confidence
            seg.id_method = m.id_method

    _atomic_write_text(
        meeting_dir / "transcript_named.json",
        json.dumps(meeting.to_dict(), indent=2),
    )

    if embeddings is not None:
        # Merge changed segment labels + embeddings — keep the caches consistent.
        try:
            _atomic_write_text(
                meeting_dir / "diarization.json",
                json.dumps([s.to_dict() for s in meeting.segments], indent=2),
            )
            emb_out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in embeddings.items()}
            _atomic_write_text(meeting_dir / "embeddings.json", json.dumps(emb_out))
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to rewrite diarization/embeddings for %s after merge", meeting_dir.name,
                exc_info=True,
            )

    try:
        from src.export import export_all
        export_all(meeting, meeting_dir / "exports")
    except Exception:
        pass  # exports regenerate at publish time; never block a save

    try:
        from src import quality
        from src.checkpoint import PipelineState
        report = quality.evaluate_meeting(meeting)
        _atomic_write_text(meeting_dir / "quality.json", json.dumps(report, indent=2))
        state = PipelineState(meeting_dir)
        state.review_status = report.get("verdict")
        state.trusted_coverage = report.get("trusted_coverage")
        state.save()
    except Exception:
        logging.getLogger(__name__).warning(
            "Gate recompute failed for %s; library badge may be stale", meeting_dir.name,
            exc_info=True,
        )


def apply_rename(meeting_id: str, label: str, new_name: str) -> bool:
    """Rename a speaker (human-authoritative) and persist. Returns False on
    unsafe/unknown meeting, unknown label, or empty name (caller maps to 404/no-op)."""
    name = (new_name or "").strip()
    if not name:
        return False
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, roster = ctx

    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False

    from src import review
    review.rename_speaker(meeting.speakers, meeting.segments, label, name, roster=roster)
    persist_review(meeting, meeting_dir)
    return True


def search_politicians_safe(q: str, *, limit: int = 10) -> dict:
    """Best-effort politician search for the link picker.

    Prefers gui.politicians (direct DB), which is the only path that can show
    which races a person is actually a candidate in — the thing a curator needs,
    because publish derives a meeting's races solely from politician_id ->
    race_candidates. Falls back to the ev-accounts HTTP search when DATABASE_URL
    isn't set, so a GUI run without DB access degrades to name + office instead
    of returning nothing. Never raises.
    """
    from gui import politicians
    if politicians.db_configured():
        return politicians.search_politicians_safe(q, limit=limit)
    return _search_politicians_http(q, limit=limit)


def _search_politicians_http(q: str, *, limit: int = 10) -> dict:
    """The pre-direct-DB path: ev-accounts /candidates/search-by-name. Carries no
    candidacy data, so `candidacy_display` is '' and the renderer omits line 2."""
    from gui import politicians
    from src.essentials_client import EssentialsClientError, search_politicians
    try:
        raw = search_politicians(q, limit=limit)
    except EssentialsClientError as exc:
        return {"results": [], "error": str(exc)}
    except Exception as exc:  # transport/unexpected — stay best-effort
        return {"results": [], "error": f"search failed: {exc}"}
    results = []
    for r in raw:
        rec = {
            "politician_slug": r.get("politician_slug") or r.get("slug"),
            "politician_id": r.get("politician_id") or r.get("id"),
            "full_name": r.get("full_name") or "",
            "office_title": r.get("office_title") or "",
            "district_label": r.get("district_label") or "",
            "government_name": r.get("government_name") or "",
            "candidacies": [],
        }
        rec["display"] = politicians.politician_display(rec)
        rec["candidacy_display"] = ""
        # False, not True: without a DB we never looked, so we must not claim the
        # person has no candidacies.
        rec["candidacy_warn"] = False
        rec["duplicate_note"] = ""
        results.append(rec)
    return {"results": results, "error": None}


def _reset_and_rename(meeting, label: str, name: str) -> None:
    """Prepare a label to take a real identity: drop any unidentified/non-speaker
    mark, then apply an optional reviewer-supplied name.

    The order of these two, and of the caller's assignment after them, is forced
    and each wrong order loses data silently:

    - clear_speaker_status blanks the placeholder name, so it must run BEFORE the
      rename, or it erases the name just supplied.
    - rename_speaker drops any prior identity when the name changes (it treats
      the old link as belonging to the old name), so it must run BEFORE the
      caller's link/local-person assignment, or it erases the identity just set.

    Hence: clear status -> rename -> assign. Both steps here no-op when there is
    nothing to do, so a plain link with no name behaves exactly as before.

    Deliberately calls rename_speaker with roster=None, never the meeting's
    roster. rename_speaker's correct_speaker_name normalisation (allow_fuzzy=
    True by default) can reassign a name to a DIFFERENT roster member whose
    surname merely resembles it — its own docstring's example is "Smithey" ->
    "...-Smith" at a 0.83 fuzzy ratio. Both callers below overwrite
    politician_* right after this returns (link_speaker sets it, assign_local_
    person clears it), so a roster-derived link here is immediately discarded
    anyway — the ONLY live effect of passing a roster would be silently
    rewriting the curator's typed name onto a roster member. That is
    especially dangerous for the local-person path, where this name is
    precisely the curator's declaration "this person is NOT on any roster",
    and where publish._upsert_local_people writes it as that person's PUBLIC
    name. So the name stored here must be exactly what the curator picked or
    typed, never a roster-normalised substitute.
    """
    from src import review

    review.clear_speaker_status(meeting.speakers, meeting.segments, label)
    if (name or "").strip():
        review.rename_speaker(meeting.speakers, meeting.segments, label,
                              name.strip(), roster=None)


def apply_link(meeting_id: str, label: str, politician_slug: str, politician_id: str,
               name: str = "") -> bool:
    """Link a speaker to an essentials politician/candidate and persist. Accepts a
    slug OR an id (candidates have an id but no slug). False on unsafe/unknown
    meeting or label, or when BOTH slug and id are empty.

    `name` is optional. The picker sends the display name of the person the
    reviewer just clicked, so the transcript's speaker_name cannot disagree with
    the linked person; callers that omit it keep the previous behaviour exactly.
    """
    slug = (politician_slug or "").strip()
    pid = (politician_id or "").strip()
    if not slug and not pid:
        return False
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    from src import review
    _reset_and_rename(meeting, label, name)
    review.link_speaker(meeting.speakers, label, slug or None, pid or None)
    persist_review(meeting, meeting_dir)
    return True


def apply_unlink(meeting_id: str, label: str) -> bool:
    """Clear a speaker's politician link and persist. False on unsafe/unknown."""
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    from src import review
    review.link_speaker(meeting.speakers, label, None, None)
    persist_review(meeting, meeting_dir)
    return True


def apply_make_local_person(meeting_id: str, label: str, slug: str, role_raw: str,
                            name: str = "") -> bool:
    """Make a speaker a site-local person and persist.

    `role_raw` is whatever the reviewer typed or picked; it goes through
    resolve_local_role, which guarantees a storable shape, so a role can never be
    invalid here. Returns False on an unsafe/unknown meeting or label. Raises
    ValueError on a slug that is malformed or already held by another label —
    a distinct failure the route reports as 400 rather than 404.

    `name` is optional but the picker always sends it, because publish writes
    `speaker_name or slug` as a local person's PUBLIC name: a nameless local
    person reaches readers as the raw slug.
    """
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    from src import review
    from src.event_kinds import resolve_local_role

    role = resolve_local_role(role_raw, meeting.event_kind)
    _reset_and_rename(meeting, label, name)
    review.assign_local_person(meeting.speakers, label, slug, role)   # may raise ValueError
    persist_review(meeting, meeting_dir)
    return True


def apply_clear_local_person(meeting_id: str, label: str) -> bool:
    """Drop a speaker's local-person identity and persist. False on unsafe/unknown
    meeting or label, and also when review.clear_local_person itself no-ops
    (no mapping to clear, or the speaker is an unidentified handle whose slug
    isn't a local person to drop) — a no-op is not success."""
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    from src import review

    if review.clear_local_person(meeting.speakers, label) is None:
        return False
    persist_review(meeting, meeting_dir)
    return True


def apply_clear_speaker_status(meeting_id: str, label: str) -> bool:
    """Undo an unidentified / non-speaker mark and persist. False on an
    unsafe/unknown meeting or label, and also when review.clear_speaker_status
    itself no-ops (the speaker was never marked) — a no-op is not success, so an
    Undo on an unmarked speaker reports 404 rather than a silent success."""
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    from src import review

    if review.clear_speaker_status(meeting.speakers, meeting.segments, label) is None:
        return False
    persist_review(meeting, meeting_dir)
    return True


def merge_voice_report(meeting_id: str, source_label: str, target_label: str) -> Optional[dict]:
    """Voice-similarity verdict for a PROPOSED merge, without performing it.

    Returns {"similarity": float|None, "verdict": str, "blocked": bool}, or None
    when the meeting or either label is unknown (the caller maps that to 404, a
    different condition from "this merge looks wrong").

    blocked is True only for a measured mismatch — an unmeasurable pair is never
    blocked, since a merge must not be refused for lack of evidence.
    """
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return None
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if source_label not in known or target_label not in known or source_label == target_label:
        return None
    from src import review
    sim = review.voice_similarity(_load_embeddings(meeting_dir), source_label, target_label)
    verdict = review.merge_voice_verdict(sim)
    return {"similarity": sim, "verdict": verdict, "blocked": verdict == "mismatch"}


def apply_merge(meeting_id: str, source_label: str, target_label: str,
                *, confirm_mismatch: bool = False) -> bool:
    """Merge source speaker into target and persist (incl. diarization+embeddings).
    False on unsafe/unknown meeting, unknown/equal labels, or merge failure.

    Also False when the two labels' voices clearly disagree and the caller has not
    passed confirm_mismatch. The merge is destructive and has no undo — it
    relabels segments and drops the source from embeddings — and a mis-merge
    leaves ONE label holding two people, which every name-based detector reads as
    clean. The route asks merge_voice_report first so it can offer a confirmation
    instead of a bare failure; this guard is the backstop for other callers.
    """
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if source_label not in known or target_label not in known or source_label == target_label:
        return False
    embeddings = _load_embeddings(meeting_dir)
    from src import review
    if not confirm_mismatch:
        sim = review.voice_similarity(embeddings, source_label, target_label)
        if review.merge_voice_verdict(sim) == "mismatch":
            return False
    try:
        review.merge_speakers(meeting.segments, embeddings, meeting.speakers, source_label, target_label)
    except ValueError:
        return False
    persist_review(meeting, meeting_dir, embeddings=embeddings)
    return True


def _mark(meeting_id: str, label: str, fn) -> bool:
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    fn(meeting, meeting_dir)
    persist_review(meeting, meeting_dir)
    return True


def apply_mark_unidentified(meeting_id: str, label: str, display_label: str = "") -> bool:
    from src import review

    def fn(meeting, meeting_dir):
        review.mark_unidentified(
            meeting.speakers, meeting.segments, label,
            meeting_dir.name, display_label=(display_label or "").strip() or None,
        )
    return _mark(meeting_id, label, fn)


def apply_mark_non_speaker(meeting_id: str, label: str, display_label: str = "") -> bool:
    from src import review

    def fn(meeting, meeting_dir):
        review.mark_non_speaker(
            meeting.speakers, meeting.segments, label,
            display_label=(display_label or "").strip() or None,
        )
    return _mark(meeting_id, label, fn)


def apply_enroll(meeting_id: str, label: str) -> bool:
    """Enroll a named speaker's voice into the profile DB (idempotent per meeting).
    False on unsafe/unknown meeting, unknown label, no name, non-speaker, or no embedding."""
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, roster = ctx
    mapping = meeting.speakers.get(label)
    if mapping is None or not (mapping.speaker_name and mapping.speaker_name.strip()):
        return False
    if getattr(mapping, "speaker_status", None) == "non_speaker":
        return False
    embeddings = _load_embeddings(meeting_dir)
    emb = embeddings.get(label)
    if emb is None:
        return False

    from src.enroll import _enroll_mapping, load_profiles, resolve_mapping_enrollment, save_profiles
    db = load_profiles()
    key, _slug, _id = resolve_mapping_enrollment(mapping, roster)
    prof = db.profiles.get(key)
    if prof is not None and meeting_dir.name in getattr(prof, "meetings_seen", []):
        return True  # already enrolled from this meeting — idempotent no-op (no duplicate record)

    seg_count = sum(1 for s in meeting.segments if s.speaker_label == label)
    _enroll_mapping(db, mapping, emb, meeting_dir.name, seg_count, roster=roster)
    save_profiles(db)
    return True


def _stale_published_warnings(meeting) -> list[dict]:
    """Warnings for live speaker rows this transcript has dropped, or [] when
    prod is clean, unpublished, or unreachable."""
    from gui import publish_api
    from src.speaker_orphans import audit_meeting, stale_publish_warnings

    rows = publish_api.published_speaker_rows(meeting.meeting_id)
    if not rows:                       # None = unknown, [] = nothing published
        return []
    audit = audit_meeting(
        meeting.meeting_id, rows,
        {"speakers": {k: v.to_dict() for k, v in meeting.speakers.items()}},
    )
    return stale_publish_warnings(audit)


def load_review_page(meeting_id: str) -> Optional[ReviewPageData]:
    if not is_safe_meeting_id(meeting_id):
        return None
    meeting_dir = config.MEETINGS_DIR / meeting_id
    named = meeting_dir / "transcript_named.json"
    if not named.exists():
        return None
    try:
        meeting = Meeting.from_dict(json.loads(named.read_text(encoding="utf-8")))
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return None

    import numpy as np
    from src.enroll import load_profiles, resolve_mapping_enrollment
    from src import review

    emb_path = meeting_dir / "embeddings.json"
    embeddings = {}
    if emb_path.exists():
        try:
            embeddings = {k: np.array(v) for k, v in json.loads(emb_path.read_text()).items()}
        except (ValueError, OSError, TypeError, AttributeError):
            embeddings = {}
    profile_db = load_profiles()
    # Load roster so enrollment keys resolve identically to apply_enroll — the
    # is_enrolled display must match what apply_enroll wrote for remapped names.
    roster = _load_roster_for(meeting_dir)

    views = review.build_review_state(
        meeting.segments, meeting.speakers, embeddings, profile_db, show_text=True
    )

    # Surface every enrollment warning (the terminal enroll flow already gets
    # these; the GUI reviewer must too) plus, per card, the peer labels that
    # share its name — a rename onto an existing name is usually a merge-in-waiting.
    warnings = review.enrollment_warnings(meeting.speakers, roster)
    # Plus the one collision enrollment_warnings structurally cannot see: a
    # meetings.speakers row for a label this transcript no longer has. Label
    # surgery rewrites the local artifact and nothing else in the GUI consults
    # prod, so dropping a label from a LIVE meeting otherwise looks finished
    # while the stale row keeps serving. Best-effort: None means unknown.
    warnings.extend(_stale_published_warnings(meeting))
    peer_labels: dict[str, list[str]] = {}
    for labels in review.duplicate_named_speakers(meeting.speakers).values():
        for lbl in labels:
            peer_labels[lbl] = [o for o in labels if o != lbl]

    # Pairwise voice similarity for the merge control: every candidate target gets
    # its own hint, and clear mismatches are named so the UI can confirm before
    # applying a merge that cannot be undone.
    labels = [v.label for v in views]
    merge_hints: dict[str, dict[str, str]] = {l: {} for l in labels}
    merge_mismatches: dict[str, list[str]] = {l: [] for l in labels}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            sim = review.voice_similarity(embeddings, a, b)
            verdict = review.merge_voice_verdict(sim)
            hint = "voice ?" if sim is None else f"voice {sim:+.2f} {_VERDICT_MARK[verdict]}"
            merge_hints[a][b] = merge_hints[b][a] = hint
            if verdict == "mismatch":
                merge_mismatches[a].append(b)
                merge_mismatches[b].append(a)

    from src.publish import extract_youtube_id, playback_for_meeting

    youtube_id = extract_youtube_id(meeting.audio_source or "")
    # HLS video source (e.g. House Clerk CDN, stored as source_audio_url). Reuse
    # the site's playback resolver so review and the live site agree on "the video".
    kind, url = playback_for_meeting(meeting)
    hls_url = url if kind == "hls" else None
    media = find_meeting_media(meeting_dir)
    media_kind = media[0] if media else None
    # Full-source playback (add clip_offset to seeks): a YouTube stream, an HLS
    # stream, or a local full-source video. Local audio (opus/wav) is clip-local.
    is_full_source = bool(youtube_id) or bool(hls_url) or (media_kind == "video")
    clip_offset = meeting.clip_start_seconds or 0.0

    confirmed: list[SpeakerCard] = []
    needs: list[SpeakerCard] = []
    for v in views:
        mapping = meeting.speakers.get(v.label)
        has_emb = v.label in embeddings
        named = bool(mapping and mapping.speaker_name and mapping.speaker_name.strip())
        not_nonspeaker = not (mapping and getattr(mapping, "speaker_status", None) == "non_speaker")
        is_enrollable = named and not_nonspeaker and has_emb
        is_enrolled = False
        profile_meetings = 0
        profile_samples = 0
        if named and not_nonspeaker:
            key, _slug, _id = resolve_mapping_enrollment(mapping, roster)
            prof = profile_db.profiles.get(key)
            if prof is not None:
                seen = getattr(prof, "meetings_seen", []) or []
                is_enrolled = meeting_dir.name in seen
                # Count only OTHER meetings until this one is enrolled, so the hint
                # shows the profile's existing strength (what enrolling would add to).
                profile_meetings = len(seen) - (1 if is_enrolled else 0)
                profile_samples = len(getattr(prof, "embeddings", []) or [])
        card = SpeakerCard(
            label=v.label,
            name=v.current_name,
            confidence=v.current_confidence,
            method=v.current_method,
            minutes=v.total_speech_seconds / 60.0,
            seg_count=v.seg_count,
            sample_text=v.sample_text,
            hints=[(h[0], h[1]) for h in v.soft_hints[:3]],
            clip_seeks=[_seek(c, is_video=is_full_source, clip_offset=clip_offset)
                        for c in v.clip_candidates],
            politician_slug=getattr(mapping, "politician_slug", None) if mapping else None,
            politician_id=getattr(mapping, "politician_id", None) if mapping else None,
            speaker_status=getattr(mapping, "speaker_status", None) if mapping else None,
            local_slug=getattr(mapping, "local_slug", None) if mapping else None,
            local_role=getattr(mapping, "local_role", None) if mapping else None,
            default_slug=review.default_local_slug(v.current_name, v.label),
            is_enrollable=is_enrollable,
            is_enrolled=is_enrolled,
            thin_sample=v.total_speech_seconds < ENROLL_MIN_SPEECH_SECONDS,
            profile_meetings=profile_meetings,
            profile_samples=profile_samples,
            duplicate_labels=peer_labels.get(v.label, []),
            merge_hints=merge_hints.get(v.label, {}),
            merge_mismatches=sorted(merge_mismatches.get(v.label, [])),
        )
        (confirmed if card.is_confirmed else needs).append(card)

    display_name = meeting.title or " ".join(
        p for p in (meeting.city, meeting.meeting_type) if p
    ) or meeting_id

    return ReviewPageData(
        meeting_id=meeting_id,
        display_name=display_name,
        media_kind=media_kind,
        youtube_id=youtube_id,
        hls_url=hls_url,
        needs_attention=needs,
        confirmed=confirmed,
        warnings=warnings,
        local_role_options=list(local_roles_for(meeting.event_kind)),
    )
