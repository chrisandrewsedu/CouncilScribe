"""GUI-facing view models. No HTTP, no I/O — pure data + display helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Friendly labels for src.checkpoint.PipelineStage values (0..7). Kept here (not
# imported from checkpoint) so the label wording is a GUI concern the pipeline
# can't accidentally change.
_STAGE_LABELS = {
    0: "Not started",
    1: "Audio ingested",
    2: "Speakers separated",
    3: "Transcribed",
    4: "Identified — ready to review",
    5: "Summarized",
    6: "Voices enrolled",
    7: "Exported",  # local export files written — NOT the same as live on the site
}


def stage_label(completed_stage: int) -> str:
    """Human label for a PipelineStage integer value."""
    return _STAGE_LABELS.get(completed_stage, f"Unknown ({completed_stage})")


def gate_badge(review_status: Optional[str], trusted_coverage: Optional[float]) -> tuple[str, str]:
    """(level, text) for the confidence-gate badge. level is a CSS class token."""
    if review_status == "pass":
        if trusted_coverage is not None:
            return "pass", f"{round(trusted_coverage * 100)}% trusted"
        return "pass", "passed"
    if review_status == "review":
        return "review", "needs review"
    if review_status == "failed":
        return "failed", "failed"
    return "none", "—"


def duration_label(seconds: Optional[float]) -> str:
    """'2h 52m' / '47m' / '—' (— for None or non-positive)."""
    if not seconds or seconds <= 0:
        return "—"
    total_minutes = int(seconds // 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


@dataclass
class MeetingSummary:
    """One row in the meeting library. Built from pipeline_state.json (+ title
    from transcript_named.json when present)."""

    meeting_id: str
    title: Optional[str]
    city: Optional[str]
    meeting_type: Optional[str]
    date: Optional[str]
    event_kind: Optional[str]
    completed_stage: int
    # Slice 1b: enrichment fields; all optional so older/partial meetings still build.
    speaker_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    review_status: Optional[str] = None
    trusted_coverage: Optional[float] = None
    has_thumbnail: bool = False
    # Live-site status from the DB: True = live, False = queried but not live,
    # None = not checked (no DB configured) so no badge is shown.
    is_live: Optional[bool] = None
    # prod's meetings.meetings.speaker_count. None = not published, or no DB.
    # Shown only when it disagrees with the local count: an orphan speaker row
    # (a label the transcript no longer has) inflates prod's, and nothing else
    # in the GUI compares the two.
    live_speaker_count: Optional[int] = None
    # Slice 3: library context. All optional so older/partial meetings still build.
    event_orgs: list = field(default_factory=list)
    body_slug: Optional[str] = None
    race_id: Optional[str] = None
    race_label: Optional[str] = None
    guest: Optional[str] = None
    processed_at: Optional[float] = None   # pipeline_state.json mtime (epoch); library sort key

    @property
    def stage_label(self) -> str:
        return stage_label(self.completed_stage)

    @property
    def processed_label(self) -> str:
        """Relative 'when last processed', from processed_at: 'just now' / '5m ago'
        / '3h ago' / a date for older. '—' when unknown."""
        if not self.processed_at:
            return "—"
        import time
        delta = time.time() - self.processed_at
        if delta < 60:
            return "just now"
        if delta < 3600:
            return f"{int(delta // 60)}m ago"
        if delta < 86400:
            return f"{int(delta // 3600)}h ago"
        import datetime
        return datetime.date.fromtimestamp(self.processed_at).isoformat()

    @property
    def live_badge(self) -> Optional[tuple[str, str]]:
        """(css_token, text) for the live-site badge, or None when unknown."""
        if self.is_live is None:
            return None
        return ("live", "Live") if self.is_live else ("notlive", "Not live")

    @property
    def speakers_label(self) -> str:
        local = "—" if self.speaker_count is None else str(self.speaker_count)
        # Any disagreement means prod is out of date, so none is special-cased:
        # too high is usually an orphan speaker row, too low a publish that
        # predates later review, and an unknown local count is the
        # not-judgeable case where prod's number is the only one there is.
        if (self.live_speaker_count is not None
                and self.live_speaker_count != self.speaker_count):
            return f"{local} (live: {self.live_speaker_count})"
        return local

    @property
    def duration_label(self) -> str:
        return duration_label(self.duration_seconds)

    @property
    def gate_badge(self) -> tuple[str, str]:
        return gate_badge(self.review_status, self.trusted_coverage)

    @property
    def display_name(self) -> str:
        """Title if set, else 'City MeetingType', else the meeting_id."""
        title = (self.title or "").strip()
        if title:
            return title
        parts = [p for p in (self.city, self.meeting_type) if p and p.strip()]
        return " ".join(parts) if parts else self.meeting_id

    @property
    def context_line(self) -> str:
        """One-line context under the row name: city · body · org(s) · race ·
        guest, de-duplicated, only what's present."""
        parts: list[str] = []
        if self.city and self.city.strip():
            parts.append(self.city.strip())
        if self.body_slug and self.body_slug.strip():
            parts.append(self.body_slug.replace("-", " ").title())
        for org in (self.event_orgs or []):
            if org and str(org).strip():
                parts.append(str(org).strip())
        if self.race_label and self.race_label.strip():
            parts.append(self.race_label.strip())
        if self.guest and self.guest.strip():
            parts.append(f"guest {self.guest.strip()}")
        seen: list[str] = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        return " · ".join(seen)

    @property
    def status_key(self) -> str:
        """Coarse lifecycle bucket for the library Status filter:
        'live' | 'ready' | 'needs-review' | 'processing'."""
        if self.is_live:
            return "live"
        if self.review_status == "pass":
            return "ready"
        if self.completed_stage >= 4:
            return "needs-review"
        return "processing"


# Confidence at/above which an identified speaker is auto-accepted (green) and
# not surfaced for attention. Mirrors the pipeline's gate threshold.
CONFIDENT_THRESHOLD = 0.85

# Below this much confirmed speech, a voice sample is too thin to enroll cleanly
# (guards against the profile pollution calibration found). Still allowed, but flagged.
ENROLL_MIN_SPEECH_SECONDS = 30.0

# A voice profile drawn from at least this many distinct meetings is "strong" —
# robust enough that enrolling another routine sample adds little. Below it, the
# profile is still "building" and clean samples are worth saving.
PROFILE_STRONG_MEETINGS = 3

_UNIDENTIFIED = "(unidentified)"


@dataclass
class SpeakerCard:
    """One speaker in the review page."""

    label: str
    name: Optional[str]
    confidence: float
    method: Optional[str]
    minutes: float
    seg_count: int
    sample_text: Optional[str] = None
    hints: list[tuple[str, float]] = field(default_factory=list)
    clip_seeks: list[float] = field(default_factory=list)
    politician_slug: Optional[str] = None
    politician_id: Optional[str] = None
    speaker_status: Optional[str] = None  # None | "unidentified" | "non_speaker"
    local_slug: Optional[str] = None
    local_role: Optional[str] = None
    default_slug: str = ""        # prefill for the make-local-person form
    is_enrollable: bool = False   # named, not a non-speaker, has an embedding
    is_enrolled: bool = False     # this meeting already contributed to the voice profile
    thin_sample: bool = False     # < ENROLL_MIN_SPEECH_SECONDS of speech
    profile_meetings: int = 0     # distinct meetings the stored voice profile draws from
    profile_samples: int = 0      # voice samples (embeddings) in the stored profile
    duplicate_labels: list[str] = field(default_factory=list)  # other labels sharing this speaker's name
    # Per-candidate merge guidance, so the merge control guides the choice BEFORE
    # the click. Keyed by the OTHER label: a short display hint, plus the labels
    # whose voice clearly differs from this one (a merge there is almost certainly
    # a mis-merge, and it has no undo).
    merge_hints: dict = field(default_factory=dict)
    merge_mismatches: list[str] = field(default_factory=list)

    @property
    def profile_strength(self) -> str:
        """'new' | 'building' | 'strong' — robustness of the existing voice profile.

        When the enroll button is shown (this meeting not yet enrolled), the counts
        reflect only OTHER meetings — i.e. how strong the profile already is before
        this one, which is exactly what tells you whether enrolling adds value."""
        if self.profile_meetings <= 0:
            return "new"
        if self.profile_meetings >= PROFILE_STRONG_MEETINGS:
            return "strong"
        return "building"

    @property
    def profile_hint(self) -> str:
        """Human label for the card, e.g. 'Profile strong — 6 samples from 4 meetings'."""
        n_m, n_s = self.profile_meetings, self.profile_samples
        if n_m <= 0:
            return "New voice — no profile yet"
        meetings = "meeting" if n_m == 1 else "meetings"
        samples = "sample" if n_s == 1 else "samples"
        word = "strong" if self.profile_strength == "strong" else "building"
        return f"Profile {word} — {n_s} {samples} from {n_m} {meetings}"

    @property
    def display_name(self) -> str:
        return self.name if self.name and self.name.strip() else _UNIDENTIFIED

    @property
    def is_linked(self) -> bool:
        return bool(self.politician_slug or self.politician_id)

    @property
    def has_local_person(self) -> bool:
        return bool(self.local_slug)

    # Reader-facing wording for each identity_kind. Separate from the kind token
    # so the template never has to spell a status out and the two can't drift.
    _IDENTITY_PILLS = {
        "roster": "roster",
        "local": "local",
        "unidentified": "unidentified",
        "non_speaker": "not a speaker",
        "none": "no identity",
    }

    @property
    def identity_kind(self) -> str:
        """'roster' | 'local' | 'unidentified' | 'non_speaker' | 'none'.

        Which of the four identity outcomes is currently in force, as one token
        the picker can switch on. Precedence is exactly src/review.identity_label's
        — status beats links, politician_* beats local_slug — so the picker can
        never disagree with what publish will store. Derived, never stored.
        """
        if self.speaker_status == "non_speaker":
            return "non_speaker"
        if self.speaker_status == "unidentified":
            return "unidentified"
        if self.politician_id or self.politician_slug:
            return "roster"
        if self.local_slug:
            return "local"
        return "none"

    @property
    def identity_pill(self) -> str:
        """Short label for the identity pill in the card head."""
        return self._IDENTITY_PILLS[self.identity_kind]

    @property
    def is_confirmed(self) -> bool:
        """A speaker counts as confirmed only when it's trusted the way the
        confidence gate means it: a real name, high confidence, AND a
        human-authoritative / trusted-tier id_method. A high-confidence
        auto-identification (auto_linked, llm, ...) is NOT confirmed — it stays in
        'needs attention' with a one-click Accept, so the review UI's 'confirmed'
        can't diverge from the gate's 'trusted coverage' (which is what let a fully
        auto-identified meeting look done yet fail the gate)."""
        from src.quality import TIER_TRUSTED, classify_method
        return (
            bool(self.name)
            and self.name.strip() not in ("", _UNIDENTIFIED)
            and self.confidence >= CONFIDENT_THRESHOLD
            and classify_method(self.method) == TIER_TRUSTED
        )

    @property
    def accept_name(self) -> Optional[str]:
        """Best one-click name to accept: the current name, else the top voice hint."""
        if self.name and self.name.strip() not in ("", _UNIDENTIFIED):
            return self.name.strip()
        if self.hints:
            return self.hints[0][0]
        return None


@dataclass
class ReviewPageData:
    meeting_id: str
    display_name: str
    media_kind: Optional[str]  # "video" | "audio" | None
    youtube_id: Optional[str] = None  # set when the source is a YouTube URL: review streams the embed
    hls_url: Optional[str] = None  # set when the source is an HLS .m3u8 (e.g. House Clerk CDN): review streams it via hls.js
    needs_attention: list[SpeakerCard] = field(default_factory=list)
    confirmed: list[SpeakerCard] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)  # review.enrollment_warnings dicts
    local_role_options: list[str] = field(default_factory=list)

    @property
    def speaker_count(self) -> int:
        return len(self.needs_attention) + len(self.confirmed)

    @property
    def all_cards(self) -> list["SpeakerCard"]:
        return self.needs_attention + self.confirmed
