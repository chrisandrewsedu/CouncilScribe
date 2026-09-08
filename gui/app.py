"""FastAPI app factory for the processing GUI.

Slice 1: a single library route. Later slices mount review/launch/publish
routers onto the same app."""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import config
from src import ingest
from src import resolve
from src.download import is_ytdlp_url

from gui import publish_api
from gui import review_api
from gui import runner
from gui import workspace
from gui.library import scan_meetings
from gui.paths import is_safe_meeting_id
from gui.review_api import find_meeting_media
from gui.runner import RunParams

_GUI_DIR = Path(__file__).resolve().parent
_templates = Jinja2Templates(directory=str(_GUI_DIR / "templates"))
# Display-only filter: 'news_clip' -> 'News Clip' in user-facing kind labels
# (values stay snake_case for form submission / filtering).
from gui.formmeta import humanize_kind as _humanize_kind
_templates.env.filters["humanize_kind"] = _humanize_kind
_REPO_DIR = _GUI_DIR.parent
_RUN_LOCAL = str(_REPO_DIR / "run_local.py")


class _NoCacheStaticFiles(StaticFiles):
    """Serve static assets with Cache-Control: no-cache so the browser always
    revalidates (via the ETag StaticFiles already sends) instead of silently
    reusing a stale JS/CSS file after an edit. Revalidation stays cheap — an
    unchanged file still returns a 304."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app() -> FastAPI:
    app = FastAPI(title="CouncilScribe GUI")
    app.mount("/static", _NoCacheStaticFiles(directory=str(_GUI_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def library(request: Request) -> HTMLResponse:
        # Read MEETINGS_DIR via the module at request time so tests that
        # monkeypatch src.config.MEETINGS_DIR are honored.
        # One batch query for live-site status; None (no DB) => no live badge.
        # One query gives both the live badge and prod's speaker_count, so the
        # Speakers column can show a divergence from the local transcript.
        live_counts = publish_api.live_meeting_speaker_counts()
        live_slugs = None if live_counts is None else set(live_counts)
        meetings = scan_meetings(config.MEETINGS_DIR, live_slugs=live_slugs,
                                 live_speaker_counts=live_counts)
        from gui import races
        race_ids = {m.race_id for m in meetings if m.race_id}
        labels = races.race_labels(race_ids) if race_ids else {}
        for m in meetings:
            if m.race_id:
                m.race_label = labels.get(m.race_id)
        from src.event_kinds import EVENT_KINDS
        from gui import batch
        bs = batch.status()
        return _templates.TemplateResponse(
            request, "library.html",
            {"meetings": meetings, "event_kinds": list(EVENT_KINDS),
             "batch_counts": bs["counts"], "batch_pending": bs["pending"]},
        )

    @app.get("/discovery", response_class=HTMLResponse)
    def discovery_page(request: Request, flash: str = "", show: str = "pending") -> HTMLResponse:
        from gui import discovery, races
        status = "deferred" if show == "deferred" else "pending"
        rows = discovery.pending_rows(status)
        labels = races.race_labels({r.race_id for r in rows if r.race_id})
        groups: dict = {}
        for r in rows:
            if r.race_id and labels.get(r.race_id):
                r.race_label = labels[r.race_id]
            groups.setdefault(r.race_label or "Unmatched", []).append(r)
        h = discovery.health()
        # health() folds the outlet-stats aggregate onto its own connection
        # (avoids a 4th DB round-trip per page load). Fall back to the
        # standalone call only for monkeypatched/legacy health dicts that
        # predate the fold and lack the key — every real call carries it.
        ostats = h.get("outlet_stats")
        if ostats is None:
            ostats = discovery.outlet_stats()
        return _templates.TemplateResponse(
            request, "discovery.html",
            {"groups": list(groups.items()), "health": h,
             "outlet_stats": ostats,
             "outletless_reviewed": h.get("outletless_reviewed", 0),
             "flash": flash, "show": status})

    def _discovery_redirect(flash: str) -> RedirectResponse:
        from urllib.parse import quote
        return RedirectResponse(url=f"/discovery?flash={quote(flash)}", status_code=303)

    @app.post("/discovery/{row_id}/approve-ingest")
    def discovery_approve_ingest(row_id: str):
        import datetime as _dt
        from gui import batch, discovery, runner
        from gui.formmeta import (DEFAULT_COMPUTE, DEFAULT_DIARIZER,
                                  FIELDS_BY_KIND, MEETING_TYPE_DEFAULTS)
        from gui.runner import RunParams
        from src.event_kinds import EVENT_KINDS

        row = discovery.get_row(row_id)
        if row is None:
            raise HTTPException(status_code=404)
        if row.status != "pending":
            return _discovery_redirect(f"already {row.status}")
        existing = runner.find_meeting_by_source(row.url)
        if existing:
            ok = discovery.set_status(row_id, "superseded",
                                      reason=f"already ingested as {existing}")
            flash = f"duplicate of {existing}"
            if not ok:
                flash += " — SAVE FAILED, retry"
            return _discovery_redirect(flash)
        from src.source_key import source_key as _source_key
        if not _source_key(row.url).startswith("youtube:"):
            ok_probe, err = discovery.probe_extractable(row.url)
            if not ok_probe:
                return _discovery_redirect(
                    f"no extractable video ({err or 'nothing found'}) — use Edit first")
        kind = row.event_kind_guess if row.event_kind_guess in EVENT_KINDS else "news_clip"
        if kind in ("community_meeting", "other") and row.race_id:
            kind = "forum"  # electoral town halls anchor to the race (domain: forum = electoral event)
        fields = FIELDS_BY_KIND.get(kind, ())
        race_id = row.race_id if "race" in fields else None
        params = RunParams(
            input=row.url,
            date=(row.published_at or "")[:10] or _dt.date.today().isoformat(),
            meeting_type=MEETING_TYPE_DEFAULTS.get(kind, "Recording"),
            event_kind=kind,
            title=row.title,
            compute=DEFAULT_COMPUTE,
            diarizer=DEFAULT_DIARIZER,
            event_orgs=[row.channel_name] if row.channel_name else [],
            race_id=race_id,
            race_slug=discovery.race_slug_for(race_id) if race_id else None,
        )
        try:
            outcome, meeting_id = batch.launch_or_enqueue(params)
        except ValueError as exc:
            return _discovery_redirect(f"error: {exc}")
        ok = discovery.set_status(row_id, "ingested")
        flash = f"{outcome}: {meeting_id or params.title}"
        if not ok:
            flash += " — SAVE FAILED, retry"
        return _discovery_redirect(flash)

    @app.post("/discovery/{row_id}/quote-source")
    def discovery_quote_source(row_id: str):
        from gui import discovery
        row = discovery.get_row(row_id)
        if row is None:
            raise HTTPException(status_code=404)
        if row.status != "pending":
            return _discovery_redirect(f"already {row.status}")
        ok = discovery.set_status(row_id, "approved")
        flash = "approved as quote source"
        if not ok:
            flash += " — SAVE FAILED, retry"
        return _discovery_redirect(flash)

    @app.post("/discovery/{row_id}/reject")
    def discovery_reject(row_id: str, reason: str = Form("other")):
        from gui import discovery
        row = discovery.get_row(row_id)
        if row is None:
            raise HTTPException(status_code=404)
        if row.status != "pending":
            return _discovery_redirect(f"already {row.status}")
        ok = discovery.set_status(row_id, "rejected", reason=reason)
        flash = "rejected"
        if not ok:
            flash += " — SAVE FAILED, retry"
        return _discovery_redirect(flash)

    @app.post("/discovery/{row_id}/watch-channel")
    def discovery_watch_channel(row_id: str):
        from gui import discovery
        row = discovery.get_row(row_id)
        if row is None:
            raise HTTPException(status_code=404)
        ok, message = discovery.watch_channel(row)
        return _discovery_redirect(message if ok else f"error: {message}")

    @app.post("/discovery/bulk")
    def discovery_bulk(action: str = Form(...),
                       row_ids: list[str] = Form(default=[]),
                       reason: str = Form("other")):
        from gui import discovery
        if not row_ids:
            return _discovery_redirect("no rows selected")
        if action == "reject":
            n = discovery.set_status_bulk(row_ids, "rejected", reason=reason)
            return _discovery_redirect(f"rejected {n}")
        if action == "restore":
            n = discovery.set_status_bulk(row_ids, "pending", reason=None)
            return _discovery_redirect(f"restored {n} to pending")
        return _discovery_redirect(f"unknown action: {action}")

    @app.get("/meetings/{meeting_id}/thumbnail")
    def thumbnail(meeting_id: str) -> FileResponse:
        if not is_safe_meeting_id(meeting_id):
            raise HTTPException(status_code=404)
        path = config.MEETINGS_DIR / meeting_id / "thumbnail.jpg"
        if not path.exists():
            raise HTTPException(status_code=404)
        return FileResponse(str(path), media_type="image/jpeg")

    @app.get("/meetings/{meeting_id}/review")
    def review_page(meeting_id: str) -> RedirectResponse:
        return RedirectResponse(url=f"/meetings/{meeting_id}?tab=review", status_code=301)

    @app.get("/meetings/{meeting_id}", response_class=HTMLResponse)
    def workspace_shell(request: Request, meeting_id: str, tab: str = "") -> HTMLResponse:
        # Pick the tab from a cheap stage read first, so we can load the review page
        # at most once and reuse it for both the panel and the header's attention count.
        stage = workspace.meeting_stage(meeting_id)
        if stage is None:
            raise HTTPException(status_code=404)
        active = tab.strip() or workspace.default_tab_for_stage(stage)
        ctx = workspace.panel_context(active, meeting_id)
        if ctx is None:  # bad ?tab value -> fall back to the default tab
            active = workspace.default_tab_for_stage(stage)
            ctx = workspace.panel_context(active, meeting_id)
        # Reuse the review page the panel already loaded so the header doesn't reload it.
        preloaded = ctx.get("page") if active == "review" else None
        attn = len(preloaded.needs_attention) if preloaded is not None else None
        header = workspace.header_context(
            meeting_id, is_live=(publish_api.meeting_published_id(meeting_id) is not None),
            attention_count=attn,
        )
        if header is None:
            raise HTTPException(status_code=404)
        return _templates.TemplateResponse(
            request, "workspace.html", {**ctx, "header": header, "active_tab": active},
        )

    @app.get("/meetings/{meeting_id}/panel/{name}", response_class=HTMLResponse)
    def workspace_panel(request: Request, meeting_id: str, name: str) -> HTMLResponse:
        ctx = workspace.panel_context(name, meeting_id)
        if ctx is None:
            raise HTTPException(status_code=404)
        return _templates.TemplateResponse(request, f"panels/{name}.html", ctx)

    @app.get("/meetings/{meeting_id}/status")
    def workspace_status(meeting_id: str) -> JSONResponse:
        st = runner.run_status(meeting_id)
        if st is None:
            raise HTTPException(status_code=404)
        # is_live changes only on an explicit publish (never mid-run), so skip the
        # remote DB round-trip while the pipeline is running to keep the poll cheap.
        is_live = None if st.get("running") else (
            publish_api.meeting_published_id(meeting_id) is not None
        )
        header = workspace.header_context(meeting_id, is_live=is_live)
        st["review_status"] = header["review_status"] if header else None
        st["is_live"] = header["is_live"] if header else None
        st["attention_count"] = header["attention_count"] if header else 0
        return JSONResponse(st)

    @app.get("/meetings/{meeting_id}/media")
    def media(meeting_id: str):
        if not is_safe_meeting_id(meeting_id):
            raise HTTPException(status_code=404)
        meeting_dir = config.MEETINGS_DIR / meeting_id
        found = find_meeting_media(meeting_dir)
        if found is None:
            raise HTTPException(status_code=404)
        _kind, filename = found
        suffix = Path(filename).suffix.lower()
        if suffix in (".opus", ".ogg"):
            media_type = "audio/ogg"
        elif suffix == ".wav":
            media_type = "audio/wav"
        else:
            media_type = "video/mp4"
        return FileResponse(str(meeting_dir / filename), media_type=media_type)

    @app.post("/meetings/{meeting_id}/cleanup")
    def cleanup_media_route(meeting_id: str):
        if not is_safe_meeting_id(meeting_id):
            raise HTTPException(status_code=404)
        from src.cleanup import cleanup_meeting

        result = cleanup_meeting(meeting_id)
        if result["status"] == "not_found":
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.post("/cleanup-all")
    def cleanup_all_route():
        from src.cleanup import backfill_all

        backfill_all()
        return RedirectResponse(url="/", status_code=303)

    @app.post("/meetings/{meeting_id}/delete")
    def delete_meeting_route(meeting_id: str, confirm_slug: str = Form("")):
        if not is_safe_meeting_id(meeting_id):
            raise HTTPException(status_code=404)
        if confirm_slug != meeting_id:
            # Typed confirmation didn't match — no-op, back to the review page.
            return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)
        from src.purge import purge_meeting

        purge_meeting(meeting_id)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/meetings/{meeting_id}/speakers/{label}/name")
    def set_speaker_name(meeting_id: str, label: str, name: str = Form("")):
        redirect = RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)
        if not name.strip():
            return redirect  # empty submission: no-op, back to the page
        if not review_api.apply_rename(meeting_id, label, name):
            raise HTTPException(status_code=404)  # unknown meeting / unsafe id / unknown label
        return redirect

    @app.get("/api/politicians/search")
    def politician_search(q: str = "") -> JSONResponse:
        return JSONResponse(review_api.search_politicians_safe(q))

    @app.get("/batch/status")
    def batch_status() -> JSONResponse:
        from gui import batch
        return JSONResponse(batch.status())

    @app.post("/batch/max")
    def batch_set_max(n: str = Form("")):
        from gui import batch
        try:
            batch.set_max_concurrent(int(n))
        except (TypeError, ValueError):
            pass
        return RedirectResponse(url="/", status_code=303)

    @app.post("/batch/pending/{pending_id}/remove")
    def batch_remove_pending(pending_id: int):
        from gui import batch
        batch.remove_pending(pending_id)
        return RedirectResponse(url="/", status_code=303)

    @app.get("/api/races/search")
    def race_search(q: str = "") -> JSONResponse:
        from gui import races
        return JSONResponse(races.search_races_safe(q))

    @app.get("/api/source-meta")
    def source_meta(url: str = "") -> JSONResponse:
        # Podcast / public-radio CMS episode pages: resolve for real metadata.
        # Looked up at call time so tests can monkeypatch resolve.resolve_source.
        try:
            resolved = resolve.resolve_source(url) if url else None
        except Exception:
            resolved = None
        if resolved is not None:
            return JSONResponse({
                "date": resolved.date,
                "title": resolved.title,
                "event_org": resolved.outlet,
            })
        # yt-dlp URLs (YouTube/Facebook): fetchable video metadata.
        if not is_ytdlp_url(url):
            return JSONResponse({"date": None, "title": None, "event_org": None})
        # Look up ingest.fetch_source_metadata at call time so tests can
        # monkeypatch it on the module.
        meta = ingest.fetch_source_metadata(url)
        return JSONResponse({
            "date": meta["upload_date"],
            "title": meta["title"],
            "event_org": meta["channel"],
        })

    @app.post("/meetings/{meeting_id}/speakers/{label}/link")
    def link_speaker_route(meeting_id: str, label: str,
                           politician_slug: str = Form(""), politician_id: str = Form(""),
                           name: str = Form("")):
        redirect = RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)
        if not politician_slug.strip() and not politician_id.strip():
            return redirect  # nothing to link
        if not review_api.apply_link(meeting_id, label, politician_slug, politician_id,
                                     name=name):
            raise HTTPException(status_code=404)
        return redirect

    @app.post("/meetings/{meeting_id}/speakers/{label}/unlink")
    def unlink_speaker_route(meeting_id: str, label: str):
        if not review_api.apply_unlink(meeting_id, label):
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.post("/meetings/{meeting_id}/speakers/{label}/merge")
    def merge_speaker_route(meeting_id: str, label: str, target: str = Form(""),
                            confirm: str = Form("")):
        # A merge is destructive and has no undo, so a pair whose voices clearly
        # disagree is bounced back unapplied unless the reviewer confirmed. That is
        # a real UI state, distinct from an unknown label (still a 404), which is
        # why the verdict is asked for BEFORE applying.
        report = review_api.merge_voice_report(meeting_id, label, target.strip())
        if report is None:
            raise HTTPException(status_code=404)  # unknown meeting/label, or self-merge
        redirect = RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)
        if report["blocked"] and not confirm.strip():
            return redirect
        if not review_api.apply_merge(meeting_id, label, target.strip(),
                                      confirm_mismatch=True):
            raise HTTPException(status_code=404)
        return redirect

    @app.post("/meetings/{meeting_id}/speakers/{label}/local-person")
    def make_local_person_route(meeting_id: str, label: str,
                               slug: str = Form(""), role: str = Form(""),
                               name: str = Form("")):
        try:
            ok = review_api.apply_make_local_person(meeting_id, label, slug, role,
                                                    name=name)
        except ValueError as exc:
            # Malformed or colliding slug. Reported, not silently ignored: the
            # form prefills a valid default, so this is a deliberate bad value.
            raise HTTPException(status_code=400, detail=str(exc))
        if not ok:
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.post("/meetings/{meeting_id}/speakers/{label}/local-person/clear")
    def clear_local_person_route(meeting_id: str, label: str):
        if not review_api.apply_clear_local_person(meeting_id, label):
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.post("/meetings/{meeting_id}/speakers/{label}/clear-status")
    def clear_speaker_status_route(meeting_id: str, label: str):
        if not review_api.apply_clear_speaker_status(meeting_id, label):
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.post("/meetings/{meeting_id}/speakers/{label}/unidentified")
    def unidentified_route(meeting_id: str, label: str, display_label: str = Form("")):
        if not review_api.apply_mark_unidentified(meeting_id, label, display_label):
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.post("/meetings/{meeting_id}/speakers/{label}/not-speaker")
    def not_speaker_route(meeting_id: str, label: str, display_label: str = Form("")):
        if not review_api.apply_mark_non_speaker(meeting_id, label, display_label):
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.post("/meetings/{meeting_id}/speakers/{label}/enroll")
    def enroll_route(meeting_id: str, label: str):
        if not review_api.apply_enroll(meeting_id, label):
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.get("/new", response_class=HTMLResponse)
    def new_meeting_form(request: Request, flash: str = "", label: str = "",
                         input: str = "", date: str = "", title: str = "",
                         event_kind: str = "", meeting_type: str = "",
                         race_id: str = "", race_slug: str = "",
                         race_label: str = "", event_orgs: str = "",
                         guest: str = "") -> HTMLResponse:
        from src.event_kinds import EVENT_KINDS
        from gui.formmeta import (EVENT_KIND_HELP, COMPUTE_HELP, DIARIZER_HELP,
                                   CITY_REQUIRED_KINDS, MEETING_TYPE_DEFAULTS,
                                   FIELDS_BY_KIND, DEFAULT_COMPUTE, DEFAULT_DIARIZER)
        from gui.rosters import list_cached_rosters
        from gui import batch
        if race_id and not race_slug:
            from gui import discovery
            race_slug = discovery.race_slug_for(race_id)
        prefill = {"input": input, "date": date, "title": title,
                   "event_kind": event_kind, "meeting_type": meeting_type,
                   "race_id": race_id, "race_slug": race_slug,
                   "race_label": race_label, "event_orgs": event_orgs,
                   "guest": guest}
        return _templates.TemplateResponse(
            request, "new_meeting.html",
            {
                "event_kinds": list(EVENT_KINDS),
                "event_kind_help": EVENT_KIND_HELP,
                "compute_help": COMPUTE_HELP,
                "diarizer_help": DIARIZER_HELP,
                "city_required_kinds": sorted(CITY_REQUIRED_KINDS),
                "meeting_type_defaults": MEETING_TYPE_DEFAULTS,
                "cached_rosters": list_cached_rosters(),
                "fields_by_kind": FIELDS_BY_KIND,
                "default_compute": DEFAULT_COMPUTE,
                "default_diarizer": DEFAULT_DIARIZER,
                "flash": flash,
                "flash_label": label,
                "batch_counts": batch.status()["counts"],
                "prefill": prefill,
            },
        )

    @app.post("/new")
    def new_meeting_launch(
        request: Request,
        input: str = Form(""),
        date: str = Form(""),
        meeting_type: str = Form(""),
        event_kind: str = Form("council"),
        city: str = Form(""),
        title: str = Form(""),
        compute: str = Form("local"),
        diarizer: str = Form("oss"),
        clip_start: str = Form(""),
        clip_end: str = Form(""),
        event_orgs: str = Form(""),
        body_slug: str = Form(""),
        crec_chamber: str = Form(""),
        guest: str = Form(""),
        race_id: str = Form(""),
        race_slug: str = Form(""),
        confirm: str = Form(""),
    ):
        if not input.strip() or not date.strip() or not meeting_type.strip():
            raise HTTPException(status_code=400, detail="input, date, and meeting_type are required")
        from gui.formmeta import CITY_REQUIRED_KINDS
        if event_kind in CITY_REQUIRED_KINDS and not city.strip():
            raise HTTPException(
                status_code=400,
                detail=f"A city is required for event kind '{event_kind}'.",
            )
        if not confirm.strip():
            existing = runner.find_meeting_by_source(input)
            if existing:
                from src.checkpoint import PipelineState
                st = PipelineState(config.MEETINGS_DIR / existing)
                return _templates.TemplateResponse(
                    request, "dedup_confirm.html",
                    {
                        "existing_id": existing,
                        "completed_stage": int(st.completed_stage),
                        "review_status": st.review_status,
                        # echo the form so "Process anyway" can resubmit with confirm=1
                        "form": {
                            "input": input, "date": date, "meeting_type": meeting_type,
                            "event_kind": event_kind, "city": city, "title": title,
                            "compute": compute, "diarizer": diarizer,
                            "clip_start": clip_start, "clip_end": clip_end,
                            "event_orgs": event_orgs, "body_slug": body_slug,
                            "crec_chamber": crec_chamber,
                            "guest": guest, "race_id": race_id, "race_slug": race_slug,
                        },
                    },
                )
        from gui.formmeta import FIELDS_BY_KIND
        _allowed = set(FIELDS_BY_KIND.get(event_kind, ()))
        if "city" not in _allowed:
            city = ""
        if "body" not in _allowed:
            body_slug = ""
        if "guest" not in _allowed:
            guest = ""
        if "race" not in _allowed:
            race_id = race_slug = ""
        if "crec_chamber" not in _allowed:
            crec_chamber = ""
        p = RunParams(
            input=input.strip(), date=date.strip(), meeting_type=meeting_type.strip(),
            event_kind=event_kind, city=city.strip() or None, title=title.strip() or None,
            compute=compute, diarizer=diarizer,
            clip_start=clip_start.strip() or None, clip_end=clip_end.strip() or None,
            event_orgs=[o.strip() for o in event_orgs.split(",") if o.strip()],
            body_slug=body_slug.strip() or None,
            crec_chamber=crec_chamber.strip() or None,
            guest=guest.strip() or None,
            race_id=race_id.strip() or None,
            race_slug=race_slug.strip() or None,
        )
        from gui import batch
        try:
            outcome, meeting_id = batch.launch_or_enqueue(p)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        label = (p.title or p.input or "").strip()
        return RedirectResponse(
            url=f"/new?flash={outcome}&label={quote(label)}", status_code=303)

    @app.get("/meetings/{meeting_id}/run")
    def run_page(meeting_id: str) -> RedirectResponse:
        return RedirectResponse(url=f"/meetings/{meeting_id}?tab=progress", status_code=301)

    @app.get("/meetings/{meeting_id}/run/status")
    def run_status_json(meeting_id: str) -> JSONResponse:
        st = runner.run_status(meeting_id)
        if st is None:
            raise HTTPException(status_code=404)
        return JSONResponse(st)

    @app.post("/meetings/{meeting_id}/redo")
    def redo_route(meeting_id: str, stage: str = Form("")):
        stage = stage.strip()
        if stage not in runner.REDO_STAGES:
            raise HTTPException(status_code=400, detail="invalid redo stage")
        if runner.launch_redo(meeting_id, stage, python_exe=sys.executable, script=_RUN_LOCAL) is None:
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/run", status_code=303)

    @app.post("/meetings/{meeting_id}/reingest")
    def reingest_route(meeting_id: str):
        if runner.launch_reingest(meeting_id, python_exe=sys.executable,
                                  script=_RUN_LOCAL) is None:
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/run", status_code=303)

    @app.post("/meetings/{meeting_id}/continue")
    def continue_route(meeting_id: str, override: str = Form("")):
        if runner.launch_resume(meeting_id, override_gate=bool(override.strip()),
                                python_exe=sys.executable, script=_RUN_LOCAL) is None:
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/run", status_code=303)

    @app.get("/meetings/{meeting_id}/edit")
    def edit_meeting_form(meeting_id: str) -> RedirectResponse:
        return RedirectResponse(url=f"/meetings/{meeting_id}?tab=details", status_code=301)

    @app.post("/meetings/{meeting_id}/edit")
    def edit_meeting_apply(
        meeting_id: str,
        title: str = Form(""), city: str = Form(""), date: str = Form(""),
        meeting_type: str = Form(""), event_kind: str = Form(""),
    ):
        fields = {"title": title, "city": city, "date": date,
                  "meeting_type": meeting_type, "event_kind": event_kind}
        if publish_api.apply_metadata_edit(meeting_id, fields) is None:
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)

    @app.get("/meetings/{meeting_id}/publish")
    def publish_confirm(meeting_id: str) -> RedirectResponse:
        return RedirectResponse(url=f"/meetings/{meeting_id}?tab=publish", status_code=301)

    @app.post("/meetings/{meeting_id}/publish", response_class=HTMLResponse)
    def publish_apply(request: Request, meeting_id: str, force: str = Form("")):
        result = publish_api.apply_publish(meeting_id, force=bool(force.strip()))
        if result.get("reason") == "unknown":
            raise HTTPException(status_code=404)
        if result.get("ok"):
            msg = (f"✓ Published · {result.get('segments', 0)} segments · "
                   f"{result.get('speakers', 0)} speakers")
            # Say so when the publish swept away rows for labels the transcript
            # no longer has — src.publish only prints this, and that print goes
            # to the uvicorn terminal, not to the reviewer's browser.
            removed = result.get("removed_speakers") or 0
            if removed:
                msg += (f" · removed {removed} stale speaker row"
                        f"{'s' if removed != 1 else ''}")
            body = f'<div class="publish-ok">{msg}</div>'
        else:
            body = (f'<div class="error-banner">Publish failed '
                    f'({result.get("reason")}): {result.get("error", "")}</div>')
        return HTMLResponse(body)

    return app
