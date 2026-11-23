/* My recordings (server/userindex.py): index your own audio / video on this Mac so the local lane can answer from it,
   timestamped ("Team sync · 12:40"). A top-bar button opens a side drawer: drop files or choose them, watch Arc
   transcribe → Arc Tagger → EmbeddingGemma embed, remove a recording (its moments leave the index).

   API: GET /api/recordings (polled while a job runs), POST /api/recordings {paths} (the Mac app passes file paths: read
   in place, never copied), POST /api/recordings/upload?name= (raw bytes: drag-drop / picker in a browser), DELETE
   /api/recordings/<id>. In the Mac app "Choose files…" asks the native side (bridge {t:"pickRecordings"}), which calls
   back HSRecordings.addPaths([...]). Esc closes; focus returns to the button. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const bridge = () => window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.onset;
  const ACCEPT = ".wav,.mp3,.m4a,.aac,.flac,.ogg,.oga,.opus,.aif,.aiff,.aifc,.caf,.mp4,.mov,.m4v";
  const STAGE = { queued: "Waiting", decoding: "Reading audio", transcribing: "Transcribing with Arc",
    tagging: "Tagging with Arc Tagger", embedding: "Embedding", uploading: "Uploading" };
  let root = null, btn = null, lastFocus = null, timer = null, recs = [], pending = [], busy = false;

  const dur = (s) => {
    if (s == null) return "";
    s = Math.round(+s);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}` : `${m}:${String(x).padStart(2, "0")}`;
  };

  function toast(msg, err) {
    const t = $("#toast");
    if (!t) return;
    t.textContent = msg;
    t.className = "toast show" + (err ? " err" : "");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { t.className = "toast"; }, 3200);
  }

  // ------------------------------------------------------------------ data
  async function refresh() {
    try {
      const r = await fetch("/api/recordings", { cache: "no-store" });
      if (!r.ok) return;
      const j = await r.json();
      recs = j.recordings || [];
      busy = !!j.busy;
    } catch { /* server restarting */ }
    render();
    schedule();
  }
  function schedule() {
    clearTimeout(timer);
    const active = busy || pending.length || recs.some((r) => !["ready", "error"].includes(r.status));
    if (active) timer = setTimeout(refresh, 900);
    if (btn) btn.classList.toggle("working", !!active);
  }

  async function addPaths(paths) {
    paths = (paths || []).filter((p) => typeof p === "string" && p);
    if (!paths.length) return;
    open();
    try {
      const r = await fetch("/api/recordings", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ paths }) });
      const j = await r.json().catch(() => ({}));
      for (const e of j.errors || []) toast(`${e.path.split("/").pop()}: ${e.error}`, true);
    } catch (e) { toast("Could not reach the server", true); }
    refresh();
  }

  async function upload(files) {
    files = Array.from(files || []);
    if (!files.length) return;
    open();
    for (const f of files) {
      const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
      if (!ACCEPT.split(",").includes(ext)) { toast(`${f.name}: not an audio or video file`, true); continue; }
      const p = { id: "up" + Math.random().toString(36).slice(2), name: f.name.replace(/\.[^.]+$/, ""), status: "uploading", progress: 0 };
      pending.push(p);
      render(); schedule();
      try {
        const r = await fetch(`/api/recordings/upload?name=${encodeURIComponent(f.name)}`, { method: "POST", body: f });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) toast(`${f.name}: ${j.error || r.status}`, true);
      } catch { toast(`${f.name}: upload failed`, true); }
      pending = pending.filter((x) => x !== p);
      await refresh();
    }
  }

  async function remove(id, name) {
    if (!window.confirm(`Remove “${name}” and its moments from the index? The original file is not touched.`)) return;
    try {
      const r = await fetch(`/api/recordings/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!r.ok) toast("Could not remove it", true);
    } catch { toast("Could not reach the server", true); }
    refresh();
  }

  function choose() {
    const b = bridge();
    if (b) { try { b.postMessage({ t: "pickRecordings" }); return; } catch { /* fall back to the file input */ } }
    $("#recFile", root).click();
  }

  // ------------------------------------------------------------------ render
  function rowHtml(r) {
    const live = !["ready", "error"].includes(r.status);
    const pct = Math.round((r.progress || 0) * 100);
    let meta;
    if (r.status === "ready") {
      const st = r.stats || {};
      meta = [dur(r.duration_s), `${r.moments} moment${r.moments === 1 ? "" : "s"}`,
        st.x_realtime ? `indexed at ${st.x_realtime}× real time` : null].filter(Boolean).join(" · ");
    } else if (r.status === "error") {
      meta = `<span class="rec-err">${esc(r.error || "failed")}</span>`;
    } else {
      meta = (r.paused ? "Paused while you ask" : (STAGE[r.status] || r.status)) + (r.status === "uploading" ? "…" : ` · ${pct}%`) +
        (r.duration_s ? ` · ${dur(r.duration_s)}` : "");
    }
    const tip = r.status === "ready" ? `Ask: “In the ${r.name}, what did …?”` : "";
    return `<li class="rec ${live ? "live" : ""} ${r.status === "error" ? "bad" : ""}" data-id="${esc(r.id)}" title="${esc(tip)}">
      <div class="rec-ic" aria-hidden="true">${r.status === "ready" ? "▶" : r.status === "error" ? "!" : ""}</div>
      <div class="rec-b"><div class="rec-n">${esc(r.name)}</div><div class="rec-m">${r.status === "error" ? meta : esc(meta)}</div>
      ${live ? `<div class="rec-bar${r.paused ? " paused" : ""}"><i style="width:${r.status === "uploading" ? 100 : Math.max(3, pct)}%"></i></div>` : ""}</div>
      ${r.status === "uploading" ? "" : `<button class="rec-x" data-rec="rm" aria-label="Remove ${esc(r.name)}" title="Remove from the index">×</button>`}</li>`;
  }

  function render() {
    if (!root) return;
    const all = [...pending, ...recs.slice().reverse()];
    const list = $("#recList", root);
    list.innerHTML = all.length ? all.map(rowHtml).join("") :
      `<li class="rec-empty">Nothing yet. Add a meeting, call or voice memo: Arc transcribes it on this Mac, Arc Tagger marks the people, places and numbers, and the local lane answers from it with a timestamp.</li>`;
    const n = recs.filter((r) => r.status === "ready").length;
    $("#recCount", root).textContent = n ? `${n} indexed` : "";
    if (btn) $("b", btn).textContent = n ? `Recordings · ${n}` : "Recordings";
  }

  function ensureRoot() {
    if (root) return root;
    root = document.createElement("div");
    root.className = "rec-dlg";
    root.hidden = true;
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-labelledby", "recTitle");
    root.innerHTML = `<div class="rec-panel">
      <div class="rec-head"><div><h2 id="recTitle">My recordings</h2>
        <p class="rec-sub">Indexed on this Mac, never uploaded. <span id="recCount"></span></p></div>
        <button class="icon-btn" data-rec="close" aria-label="Close">×</button></div>
      <div class="rec-drop" id="recDrop" tabindex="0" role="button" aria-label="Add recordings">
        <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M11 16V7.83l-3.59 3.58L6 10l6-6 6 6-1.41 1.41L13 7.83V16h-2Zm-6 4v-5h2v3h10v-3h2v5H5Z"/></svg>
        <b>Drop audio or video here</b><span>or <u>choose files…</u> · wav, mp3, m4a, aac, flac, ogg, mp4, mov</span>
        <input type="file" id="recFile" accept="${ACCEPT}" multiple hidden></div>
      <ul class="rec-list" id="recList"></ul>
      <p class="rec-fine">Ask about a recording by its name (“in the team sync, …”) or say “in my recordings”. Answers cite the moment: <span class="rec-eg">▶ 12:40</span>. Indexing pauses while you ask a question.</p>
    </div>`;
    document.body.appendChild(root);
    root.addEventListener("click", (e) => {
      if (e.target === root) return close();
      const b = e.target.closest("[data-rec]");
      if (b && b.dataset.rec === "close") return close();
      if (b && b.dataset.rec === "rm") {
        const li = b.closest("li");
        const r = recs.find((x) => x.id === li.dataset.id);
        if (r) remove(r.id, r.name);
        return;
      }
      if (e.target.closest("#recDrop") && !e.target.closest("input")) choose();
    });
    const drop = $("#recDrop", root);
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } });
    $("#recFile", root).addEventListener("change", (e) => { upload(e.target.files); e.target.value = ""; });
    root.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.stopPropagation(); close(); }
      if (e.key === "Tab") {                      // keep focus inside the drawer
        const f = Array.from(root.querySelectorAll("button, [tabindex='0']")).filter((x) => !x.hidden);
        if (!f.length) return;
        const i = f.indexOf(document.activeElement);
        if (e.shiftKey && i <= 0) { e.preventDefault(); f.at(-1).focus(); } else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
      }
      e.stopPropagation();                        // Space / digits here are not push-to-talk or replay shortcuts
    });
    return root;
  }

  function open() {
    ensureRoot();
    if (!root.hidden) return;
    lastFocus = document.activeElement;
    root.hidden = false;
    document.body.classList.add("rec-on");
    $("#recDrop", root).focus();
    refresh();
  }
  function close() {
    if (!root || root.hidden) return;
    root.hidden = true;
    document.body.classList.remove("rec-on");
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  // files dropped anywhere on the window: index them (and never let the web view navigate to the file)
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; document.body.classList.add("rec-dragging"); });
  window.addEventListener("dragover", (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
  window.addEventListener("dragleave", (e) => { if (!hasFiles(e)) return; if (--depth <= 0) { depth = 0; document.body.classList.remove("rec-dragging"); } });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    document.body.classList.remove("rec-dragging");
    upload(e.dataTransfer.files);
  });

  function init() {
    const controls = $(".top .controls");
    if (!controls) return;
    btn = document.createElement("button");
    btn.className = "sc-open rec-open";
    btn.type = "button";
    btn.id = "recOpen";
    btn.setAttribute("aria-haspopup", "dialog");
    btn.title = "Index your own recordings: the local lane answers from them, with timestamps";
    btn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M12 14a3 3 0 0 0 3-3V5a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Zm-7-3h2a5 5 0 0 0 10 0h2a7 7 0 0 1-6 6.92V21h-2v-3.08A7 7 0 0 1 5 11Z" opacity=".35"/><path fill="currentColor" d="M3 4h2v16H3zM19 4h2v16h-2z"/></svg><b>Recordings</b>`;
    const sc = $("#scOpen");
    controls.insertBefore(btn, sc || null);
    btn.addEventListener("click", open);
    ensureRoot();
    refresh();
  }

  window.HSRecordings = { open, close, addPaths, upload, refresh };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
