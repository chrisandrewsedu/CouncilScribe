// One-pane meeting workspace: tab swapping (no reload), live status polling, and
// form interception (POST then re-fetch the active panel). Absorbs run.js (status
// stepper/log) and review.js (clip seek, link search, HLS attach).
(function () {
  const header = document.querySelector(".ws-header");
  if (!header) return;
  const id = header.getAttribute("data-meeting-id");
  const panel = document.getElementById("panel");
  let activeTab = header.getAttribute("data-active-tab");

  const enc = encodeURIComponent;

  // ---- Panel loading ------------------------------------------------------
  async function loadPanel(tab, push) {
    try {
      const resp = await fetch(`/meetings/${enc(id)}/panel/${enc(tab)}`);
      if (!resp.ok) return;
      panel.innerHTML = await resp.text();
    } catch (_) { return; }
    activeTab = tab;
    header.setAttribute("data-active-tab", tab);
    header.querySelectorAll(".tabstrip .tab").forEach((a) =>
      a.classList.toggle("active", a.getAttribute("data-tab") === tab));
    if (push) history.pushState({ tab }, "", `/meetings/${enc(id)}?tab=${enc(tab)}`);
    initPanel();
    refreshStatus();
  }

  header.addEventListener("click", (e) => {
    const a = e.target.closest(".tabstrip .tab");
    if (!a) return;
    e.preventDefault();
    loadPanel(a.getAttribute("data-tab"), true);
  });

  window.addEventListener("popstate", (e) => {
    const tab = (e.state && e.state.tab) || new URLSearchParams(location.search).get("tab") || activeTab;
    loadPanel(tab, false);
  });

  // ---- Form interception --------------------------------------------------
  // In-panel forms POST via fetch, then the active panel is re-fetched. Forms
  // marked data-navigate (kebab Clean up / Delete) submit normally (full nav).
  document.addEventListener("submit", async (e) => {
    const form = e.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (form.hasAttribute("data-navigate")) return;         // let it navigate
    if (!panel.contains(form)) return;                       // only in-panel forms
    e.preventDefault();
    // The publish form returns a result fragment (✓ Published … / error); every
    // other action 303-redirects and we just re-fetch. Keep the publish result so
    // it can be shown in the panel's #publish-result slot after the re-render.
    const isPublish = form.matches(".publish-form");
    const body = new FormData(form);
    // A merge cannot be undone — it relabels every segment and drops one voice
    // profile — and a mis-merge leaves ONE label holding two people, which every
    // name-based detector then reads as clean. The server refuses an unconfirmed
    // mismatch anyway; this turns that refusal into a decision instead of a
    // silent no-op.
    const mismatches = (form.getAttribute("data-merge-mismatch") || "")
      .split(",").filter(Boolean);
    if (mismatches.length) {
      const target = (body.get("target") || "").toString();
      if (target && mismatches.includes(target)) {
        const from = form.action.split("/speakers/")[1].split("/")[0];
        if (!window.confirm(
              `${from} and ${target} do not sound like the same person.\n\n` +
              `Merging is destructive and cannot be undone. Merge anyway?`)) return;
        body.append("confirm", "1");
      }
    }
    let publishResult = "";
    try {
      const r = await fetch(form.action, { method: "POST", body, redirect: "manual" });
      if (isPublish) publishResult = await r.text();
    } catch (_) { /* best-effort; re-fetch shows current state */ }
    await loadPanel(activeTab, false);
    if (isPublish) {
      const slot = document.getElementById("publish-result");
      if (slot) slot.innerHTML = publishResult;
    }
  });

  // ---- Live status --------------------------------------------------------
  async function refreshStatus() {
    let st;
    try {
      const resp = await fetch(`/meetings/${enc(id)}/status`);
      if (!resp.ok) return;
      st = await resp.json();
    } catch (_) { return; }

    // Header pills.
    const gate = document.getElementById("gate-pill");
    if (gate && st.review_status) {
      gate.textContent = st.review_status === "pass" ? "passed"
        : st.review_status === "review" ? "needs review"
        : st.review_status === "failed" ? "failed" : "—";
    }
    const live = document.getElementById("live-pill");
    if (live && st.is_live != null) {
      live.textContent = st.is_live ? "Live" : "Not live";
      live.className = "live-badge live-" + (st.is_live ? "live" : "notlive");
    }
    const dot = document.getElementById("attn-dot");
    if (dot) dot.hidden = !st.attention_count;  // element is always present; toggle it live

    // Progress panel (if shown): update stepper + log in place, poll while running.
    const stepper = document.getElementById("stepper");
    if (stepper) {
      const logEl = document.getElementById("log");
      if (logEl && st.log_tail) { logEl.textContent = st.log_tail; logEl.scrollTop = logEl.scrollHeight; }
      stepper.querySelectorAll("li").forEach((li) => {
        const s = parseInt(li.getAttribute("data-stage"), 10);
        li.classList.toggle("done", s <= st.completed_stage);
        li.classList.toggle("current", s === st.completed_stage + 1 && st.running);
      });
      const err = document.getElementById("error-banner");
      if (err && st.exit_code != null && st.exit_code !== 0) {
        err.hidden = false;
        err.textContent = `Process exited with code ${st.exit_code}. See log below.`;
      }
      if (st.running) setTimeout(refreshStatus, 1500);
    }
  }

  // ---- Per-panel init (clip seek, link search, HLS attach) ----------------
  function initPanel() {
    attachHls();
  }

  // Clip seek: click a .clip button to seek the media (YouTube iframe or <video>/<audio>).
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".clip");
    if (!btn) return;
    const seek = parseFloat(btn.getAttribute("data-seek"));
    if (Number.isNaN(seek)) return;
    const yt = document.getElementById("yt-player");
    if (yt) { yt.src = yt.src.split("?")[0] + "?start=" + Math.floor(seek) + "&autoplay=1"; return; }
    const player = document.getElementById("player");
    if (!player) return;
    player.currentTime = seek;
    player.play();
  });

  // Politician link search: debounced query; each result is a native POST form
  // (intercepted by the submit handler above, so linking refreshes the panel).
  const DEBOUNCE = 250;
  document.addEventListener("input", (e) => {
    const input = e.target;
    if (!input.matches(".link-search input")) return;
    const widget = input.closest(".link-search");
    const results = widget.querySelector(".link-results");
    const q = input.value.trim();
    clearTimeout(widget._t);
    if (q.length < 2) { results.innerHTML = ""; return; }
    widget._t = setTimeout(async () => {
      const url = (widget.getAttribute("data-search-url") || "/api/politicians/search") + "?q=" + enc(q);
      let data;
      try { data = await (await fetch(url)).json(); }
      catch (_) { results.innerHTML = '<div class="link-msg">search unavailable</div>'; return; }
      const list = data.results || [];
      if (data.error || !list.length) {
        results.innerHTML = '<div class="link-msg">' + (data.error ? "search unavailable" : "no matches") + "</div>";
        return;
      }
      let action = widget.getAttribute("data-link-action") || "";
      if (!action.endsWith("/link")) action += "/link";
      const esc = (s) => String(s == null ? "" : s).replace(/"/g, "&quot;").replace(/</g, "&lt;");
      results.innerHTML = list.map((r) => {
        // The server composes both lines: it's the only side that knows the
        // candidacy data, and a wrong pick here silently detaches the meeting
        // from its race (publish derives races from politician_id alone).
        const cand = r.candidacy_display || "";
        const warn = !!r.candidacy_warn;
        let inner = '<span class="pr-name">' + esc(r.display || r.full_name) + "</span>";
        if (r.duplicate_note) {
          inner += '<span class="pr-warn pr-dupe">' + esc(r.duplicate_note) + "</span>";
        }
        if (cand) {
          inner += '<span class="pr-cand' + (warn ? " pr-warn" : "") + '">' + esc(cand) + "</span>";
        }
        return (
          '<form method="post" action="' + action + '">' +
          '<input type="hidden" name="politician_slug" value="' + esc(r.politician_slug) + '">' +
          '<input type="hidden" name="politician_id" value="' + esc(r.politician_id) + '">' +
          // Carry the name the reviewer just clicked, so the transcript's
          // speaker_name cannot disagree with the person linked. Must NOT fall
          // back to r.display: that's politician_display(rec), a decorated
          // "Name · Office · District · Government" line, never a bare name —
          // if full_name is empty (the no-DB HTTP fallback can omit it), an
          // empty name is correct (apply_link/_reset_and_rename just skips the
          // rename), never the whole decorated line landing in speaker_name.
          '<input type="hidden" name="name" value="' + esc(r.full_name || "") + '">' +
          '<button type="submit" class="link-result">' + inner + "</button></form>"
        );
      }).join("");
    }, DEBOUNCE);
  });

  // Identity chooser: reveal the panel for the chosen outcome. Delegated on
  // document, so it survives a panel re-fetch with no re-init — the server
  // already rendered the current outcome checked and its panel un-hidden, so
  // this only handles switching.
  document.addEventListener("change", (e) => {
    const radio = e.target;
    if (!(radio instanceof HTMLInputElement) || radio.type !== "radio") return;
    if (!radio.name.startsWith("ident-")) return;
    const block = radio.closest(".ident");
    if (!block) return;
    block.querySelectorAll(".ident-panel").forEach((p) => {
      p.hidden = p.getAttribute("data-ident") !== radio.value;
    });
  });

  // HLS attach: <video id="player" data-hls="..."> with no src. Prefer hls.js
  // (Chrome/Firefox/Edge/desktop Safari can't play HLS natively despite a truthy
  // canPlayType), fall back to native only for iOS/older Safari.
  function attachHls() {
    const video = document.getElementById("player");
    if (!video) return;
    const src = video.getAttribute("data-hls");
    if (!src || video._hlsAttached) return;
    video._hlsAttached = true;
    const useNative = () => { if (video.canPlayType("application/vnd.apple.mpegurl")) video.src = src; };
    const script = document.createElement("script");
    script.src = "/static/hls.min.js";
    script.onload = () => {
      if (window.Hls && window.Hls.isSupported()) {
        const hls = new window.Hls(); hls.loadSource(src); hls.attachMedia(video);
      } else { useNative(); }
    };
    script.onerror = useNative;
    document.head.appendChild(script);
  }

  // Initial paint: the shell server-rendered the active panel, so just wire it up.
  initPanel();
  refreshStatus();
})();
