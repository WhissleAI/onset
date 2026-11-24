// Onset: the consumer page (web/index.html). One conversation: what you asked, the answer, where it came from.
// The engine is the same as the developer view (web/lab.html: race, timeline, query frame); this page runs Onset only
// (modes ["headstart"]) and shows none of the machinery. window.__hs keeps the automation contract the Mac app's
// release smoke uses (macapp/Sources/Automation.swift): ready, run.running, run.q.answer, replay(f), mic.*, #typeBox/#typeForm.
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const MODE = "headstart", MODES = [MODE];
  // the app shell: Mac = WKWebView's window.webkit.messageHandlers.onset; Windows = pywebview's window.pywebview.api
  // (winapp/onset_win.py Api.message). In a plain browser there is none (no Settings button, files don't open).
  const bridge = () => {
    if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.onset) return window.webkit.messageHandlers.onset;
    if (window.pywebview && window.pywebview.api && window.pywebview.api.message) return { postMessage: (m) => window.pywebview.api.message(m) };
    return null;
  };
  const post = (m) => { const b = bridge(); if (!b) return false; try { b.postMessage(m); return true; } catch { return false; } };

  // ------------------------------------------------------------------ errors → the Mac app's log
  let errsSent = 0;
  const reportError = (msg) => { if (errsSent < 20 && post({ t: "log", msg: String(msg).slice(0, 400) })) errsSent++; };
  addEventListener("error", (e) => reportError(`error: ${e.message} at ${String(e.filename || "").split("/").pop()}:${e.lineno}`));
  addEventListener("unhandledrejection", (e) => reportError(`unhandled rejection: ${e.reason && (e.reason.message || e.reason)}`));

  window.__hsNotice = (msg) => {
    let n = document.getElementById("pageNotice");
    if (!n) {
      n = document.createElement("div");
      n.id = "pageNotice"; n.className = "page-notice"; n.setAttribute("role", "status");
      n.innerHTML = '<span></span><button type="button">Dismiss</button>';
      n.querySelector("button").addEventListener("click", () => { n.hidden = true; });
      document.body.appendChild(n);
    }
    n.querySelector("span").textContent = String(msg || "");
    n.hidden = false;
  };
  let toastT = null;
  function toast(msg) {
    let t = document.getElementById("toast");
    if (!t) { t = document.createElement("div"); t.id = "toast"; t.className = "toast"; t.setAttribute("role", "status"); document.body.appendChild(t); }
    t.textContent = msg; t.classList.add("on");
    clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("on"), 2600);
  }

  // ------------------------------------------------------------------ connection + health
  let ws = null, wsOpen = false, warm = false, health = null, ridSeq = 0, run = null, fresh = false;
  function connect() {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { wsOpen = true; paintStatus(); };
    ws.onclose = () => {
      wsOpen = false; paintStatus();
      if (mic.on) stopMic(false, "socket closed");
      if (run) run.running = false;
      setTimeout(connect, 1500);
    };
    ws.onmessage = (m) => {
      let e; try { e = JSON.parse(m.data); } catch { return; }
      try { onEvent(e); } catch (err) { console.error("event", e, err); reportError("event: " + err.message); }
    };
  }
  function send(obj) {
    if (!ws || ws.readyState !== 1) { toast("Onset is still starting. Try again in a moment."); return false; }
    if (obj.t === "start" || obj.t === "replay") {
      obj = { ...obj, rid: ++ridSeq };
      if (fresh) { obj.fresh = true; fresh = false; }
    }
    ws.send(JSON.stringify(obj));
    return true;
  }
  function paintStatus() {
    const s = $("#status");
    let cls = "", txt = "";
    if (!wsOpen) { cls = "down"; txt = "Reconnecting…"; }
    else if (!warm) { cls = "cold"; txt = "Getting ready…"; }
    s.hidden = !txt;
    s.className = "status " + cls;
    s.querySelector("b").textContent = txt;
    paintNotice();
  }
  function paintNotice() {
    const b = $("#notice");
    // only what the user can act on or should know about; the rest stays in the logs
    const list = wsOpen && warm && health && Array.isArray(health.notices) ? health.notices.filter((n) => n.level !== "info") : [];
    const n = list[0];
    b.hidden = !n;
    if (!n) return;
    const act = n.action === "settings" && bridge() ? `<button type="button" data-act="settings">${esc(n.action_label || "Open Settings")}</button>` : "";
    b.innerHTML = esc(String(n.text || "").replace(/\s·\s/g, ". ")) + act;
  }
  $("#notice").addEventListener("click", (e) => { if (e.target.closest("[data-act=settings]")) post({ t: "settings" }); });
  async function pollHealth() {
    try { health = await (await fetch("/api/health", { cache: "no-store" })).json(); warm = !!health.warm; }
    catch { warm = false; }
    paintStatus();
    setTimeout(pollHealth, warm ? 15000 : 2000);
  }

  // ------------------------------------------------------------------ the conversation
  const thread = $("#thread");
  const turns = new Map();          // question id (server) -> turn
  let cur = null;                   // the turn being asked / answered now

  function newTurn(qid) {
    $("#hello") && ($("#hello").hidden = true);
    $("#newChat").hidden = false;
    const el = document.createElement("article");
    el.className = "turn";
    el.innerHTML = '<h2 class="q"></h2><div class="a"></div><div class="files" hidden></div><div class="meta" hidden></div>';
    thread.appendChild(el);
    const t = { el, qid, rid: run && run.rid, words: [], mask: new Set(), text: "", answer: "", docs: [], state: "asking", band: null, ready: null, more: false };
    if (qid != null) turns.set(qid, t);
    cur = t;
    if (run) run.q = { answer: {} };
    paintTurn(t);
    return t;
  }
  // question ids restart with every run (each typed question is its own run): key turns by run + question
  function turnFor(e) {
    const k = e.q != null && run ? `${run.rid}:${e.q}` : null;
    if (k != null && turns.has(k)) return turns.get(k);
    if (cur && cur.rid === (run && run.rid) && (cur.qid == null || k == null)) {
      if (k != null && cur.qid == null) { cur.qid = k; turns.set(k, cur); }
      return cur;
    }
    return newTurn(k);
  }
  function dropTurn(t) {
    t.el.remove();
    if (t.qid != null) turns.delete(t.qid);
    if (cur === t) cur = null;
    if (!thread.querySelector(".turn")) { $("#hello").hidden = false; $("#newChat").hidden = true; }
  }

  const host = (u) => { try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return ""; } };
  const deviceOf = (d) => d.device || (d.meta && d.meta.device) || null;
  const SOURCE_NAME = { wikidata: "Wikidata", wikipedia: "Wikipedia", "open-meteo": "Open-Meteo", weather: "Open-Meteo" };
  function srcLabel(d) {
    if (d.url && /^https?:/.test(d.url)) return host(d.url);
    const s = String(d.source || d.lane || "").toLowerCase();
    if (/local|this mac|unit/.test(s)) return "On this Mac";
    return SOURCE_NAME[s] || (d.source ? String(d.source) : "Onset");
  }
  function questionHtml(t) {
    if (t.words.length) {
      return t.words.map((w, i) => t.mask.has(i) ? `<span class="masked">${esc(w.w[0] || "")}${"*".repeat(Math.max(2, w.w.length - 1))}</span>` : esc(w.w)).join(" ");
    }
    return esc(t.text);
  }
  function answerHtml(t) {
    // inline [n] markers and light markdown are dropped: the source pills under the answer say where it came from
    return esc(String(t.answer).replace(/\*\*(.+?)\*\*/g, "$1").replace(/^#+\s*/gm, "").replace(/\s*\[\d{1,2}\](?:\[\d{1,2}\])*/g, ""));
  }
  // An answer that is only a web page's title (or the top source sentence because the writer was unavailable) is not
  // an answer: show the top results instead, as a search page would
  const normT = (x) => String(x || "").toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
  function weakAnswer(t) {
    if (t.state !== "done" || !t.answer) return false;
    const web = t.docs.filter((d) => d.url && /^https?:/.test(d.url) && !deviceOf(d));
    if (!web.length || t.docs.some((d) => d.card || (d.meta && d.meta.card))) return false;
    const a = normT(t.answer.replace(/\s*\[\d+\]/g, ""));
    if (web.some((d) => { const ti = normT(d.title); return ti && (ti === a || (a.length > 12 && ti.startsWith(a)) || (ti.length > 12 && a.startsWith(ti) && a.length < ti.length + 12)); })) return true;
    return t.fallback === "extractive" && /^(what|which) (are|is) (some|the best|good)|\b(best|good|top|recommend)/.test(normT(t.text || t.words.map((w) => w.w).join(" ")));
  }
  function resultsHtml(t) {
    const web = t.docs.filter((d) => d.url && /^https?:/.test(d.url) && !deviceOf(d)).slice(0, 4);
    return '<p class="lead">Here is what I found:</p><ol class="results">' + web.map((d) =>
      `<li><a href="${esc(d.url)}" target="_blank" rel="noopener"><b>${esc(String(d.title || host(d.url)).split(/ [|·–-] /)[0])}</b></a>` +
      `<small>${esc(host(d.url))}${d.text ? " · " + esc(String(d.text).replace(/\s+/g, " ").slice(0, 140)) + (String(d.text).length > 140 ? "…" : "") : ""}</small></li>`).join("") + "</ol>";
  }
  function paintTurn(t) {
    const q = t.el.querySelector(".q"), a = t.el.querySelector(".a"), meta = t.el.querySelector(".meta"), files = t.el.querySelector(".files");
    q.innerHTML = questionHtml(t) || '<span class="interim">Listening…</span>';
    q.classList.toggle("live", t.state === "asking");
    a.classList.toggle("muted", t.state === "error");
    const weak = weakAnswer(t);
    if (weak) a.innerHTML = resultsHtml(t);
    else if (t.answer) a.innerHTML = answerHtml(t);
    else if (t.state === "error") a.textContent = "Couldn't get an answer right now. Try again in a moment.";
    else if (t.state === "thinking") a.innerHTML = '<span class="thinking" aria-label="Thinking"><i></i><i></i><i></i></span>';
    else if (t.state === "done") a.textContent = t.docs.length ? "" : "I couldn't find an answer to that.";
    else a.innerHTML = "";

    // "find my …": the device card's own list keeps Spotlight's order (the ranker may trim the per-file rows)
    const card = t.docs.map((d) => d.card).find((c) => c && c.data && c.data.kind === "files");
    const rows = card ? (card.data.files || []) : t.docs.filter((d) => { const v = deviceOf(d); return v && v.kind === "file" && v.path; })
      .map((d) => ({ path: deviceOf(d).path, name: d.title, where: d.text || "" }));
    files.hidden = !rows.length;
    files.innerHTML = rows.map((f) => {
      const name = f.name || f.path.split("/").pop(), ext = (name.match(/\.(\w{1,4})$/) || [, ""])[1];
      const sub = [f.where, f.modified].filter(Boolean).join(" · ") || f.path.replace(/^\/Users\/[^/]+/, "~");
      return `<button type="button" class="file" data-path="${esc(f.path)}"><span class="ico">${esc(ext || "file")}</span>` +
        `<span><b>${esc(name)}</b><small>${esc(sub)}</small></span></button>`;
    }).join("");

    const web = t.docs.filter((d) => !deviceOf(d));
    const seen = new Set(), pills = [];
    for (const d of web) { const k = srcLabel(d); if (!seen.has(k)) { seen.add(k); pills.push(d); } }
    const shown = weak ? [] : t.more ? pills : pills.slice(0, 3);
    let html = shown.map((d) => {
      const lab = srcLabel(d), tag = d.url && /^https?:/.test(d.url) ? "a" : "span";
      const href = tag === "a" ? ` href="${esc(d.url)}" target="_blank" rel="noopener"` : "";
      return `<${tag} class="src"${href} title="${esc(d.title || lab)}"><i class="fi" style="background:${hue(lab)}"></i><span>${esc(lab)}</span></${tag}>`;
    }).join("");
    if (!weak && pills.length > shown.length) html += `<button type="button" class="more-src">+${pills.length - shown.length} more</button>`;
    if (t.state === "done" && t.band === "low" && t.answer && !weak) html += '<span class="unsure">Not sure. Check the sources.</span>';
    if (t.state === "done" && t.ready != null && t.answer) html += `<span class="speed" title="How long after you stopped talking the answer started">${fmtReady(t.ready)}</span>`;
    meta.hidden = !html;
    meta.innerHTML = html;
  }
  const fmtReady = (ms) => (ms <= 150 ? "Ready as you finished" : `${(ms / 1000).toFixed(1)} s`);
  function hue(s) { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360; return `hsl(${h} 55% 62%)`; }
  thread.addEventListener("click", (e) => {
    const f = e.target.closest(".file");
    if (f) { if (!post({ t: "open", path: f.dataset.path })) toast("Open files from the Onset app"); return; }
    const m = e.target.closest(".more-src");
    if (m) { for (const t of turns.values()) if (t.el.contains(m)) { t.more = true; paintTurn(t); } if (cur && cur.el.contains(m)) { cur.more = true; paintTurn(cur); } }
  });
  const follow = () => { thread.scrollTop = thread.scrollHeight; };

  // ------------------------------------------------------------------ server events
  function onEvent(e) {
    if (e.mode === "sys") {
      if (e.t === "ready") { if (e.warm !== undefined) warm = !!e.warm; paintStatus(); }
      else if (e.t === "run_done" && run && (e.rid == null || e.rid === run.rid)) run.running = false;
      else if (e.t === "error") toast("Something went wrong. Try again.");
      return;
    }
    if (e.rid != null && (!run || e.rid !== run.rid)) return;
    if (e.mode !== MODE) return;
    let t;
    switch (e.t) {
      case "words": {
        if (!e.words || !e.words.length) return;
        t = turnFor(e);
        t.words = e.words;
        if (t.state === "done" || t.state === "error") return;
        paintTurn(t); follow();
        break;
      }
      case "frame":
        t = turnFor(e);
        t.mask = new Set(e.mask || []);
        for (const [i] of e.mask_kind || []) t.mask.add(i);
        paintTurn(t);
        break;
      case "heard":                       // a statement or noise, not a question: no turn for it
        t = turnFor(e);
        if (!t.answer) dropTurn(t);
        break;
      case "endpoint":
        t = turnFor(e);
        t.state = "thinking"; paintTurn(t); follow();
        break;
      case "sources":
        t = turnFor(e);
        t.docs = e.docs || []; paintTurn(t);
        break;
      case "answer":
        t = turnFor(e);
        t.answer += e.delta || "";
        if (t.state !== "done") t.state = "answering";
        if (run && t === cur) run.q.answer[MODE] = t.answer;
        paintTurn(t); follow();
        break;
      case "answer_error":
        t = turnFor(e);
        if (!t.answer) t.state = "error";
        paintTurn(t);
        break;
      case "extended":                    // the speaker kept going or corrected themselves: one new answer follows
        t = turnFor(e);
        Object.assign(t, { answer: "", docs: [], state: "asking", band: null, ready: null });
        paintTurn(t);
        break;
      case "confidence":
        t = turnFor(e);
        t.band = e.band || null; paintTurn(t);
        break;
      case "metric":
        t = turnFor(e);
        t.state = "done";
        if (!t.words.length && e.question) t.text = e.question;
        if (!t.answer && e.answer) t.answer = e.answer;
        t.fallback = e.answer_fallback || null;
        t.ready = typeof e.ready_ms === "number" ? Math.max(0, e.ready_ms) : null;
        if (run && t === cur) run.q.answer[MODE] = t.answer;
        paintTurn(t); follow();
        break;
    }
  }

  function startRun(kind) {
    run = { rid: ridSeq, kind, running: true, q: { answer: {} } };
  }

  // ------------------------------------------------------------------ typing
  const box = $("#typeBox");
  const typed = { open: false, rid: null, last: "" };
  function typeSend(final) {
    const text = box.value;
    if (!typed.open) {
      if (!text.trim()) return;
      if (mic.on) stopMic(true, "typing");
      if (!send({ t: "start", typed: true, modes: MODES })) return;
      startRun("typed");
      Object.assign(typed, { open: true, rid: ridSeq, last: "" });
      const t = newTurn(null);
      t.text = text;
    }
    if (text === typed.last && !final) return;
    typed.last = text;
    if (cur && !cur.words.length) { cur.text = text; paintTurn(cur); }
    send({ t: "type", rid: typed.rid, text, final: !!final });
    if (final) { Object.assign(typed, { open: false, last: "" }); box.value = ""; follow(); }
  }
  box.addEventListener("input", () => typeSend(false));
  box.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.isComposing || e.keyCode === 229)) e.preventDefault(); });
  $("#typeForm").addEventListener("submit", (e) => { e.preventDefault(); if (box.value.trim() || typed.open) typeSend(true); });
  $("#try").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    box.value = b.textContent; typeSend(false); typeSend(true);
  });

  // ------------------------------------------------------------------ microphone (16 kHz Float32, 100 ms frames)
  const MIC_KEEP_MS = 60000;
  const mic = { on: false, ctx: null, stream: null, src: null, node: null, sink: null, buf: [], acc: 0, cnt: 0, frac: 0, ratio: 3, level: 0, idle: null, st: null, kind: "" };
  const WORKLET = `class Cap extends AudioWorkletProcessor{process(i){const c=i[0]&&i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('hs-cap',Cap);`;
  const newStats = () => ({ t0: performance.now(), frames: 0, inS: 0, sent: 0, mutes: 0, reused: 0 });
  const captureLive = () => !!(mic.stream && mic.ctx && mic.ctx.state !== "closed" && mic.stream.getAudioTracks().some((t) => t.readyState === "live"));
  async function openCapture() {
    if (captureLive()) { if (mic.ctx.state !== "running") { try { await mic.ctx.resume(); } catch { /* stats */ } } return true; }
    releaseCapture();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream), sink = ctx.createGain();
    sink.gain.value = 0; sink.connect(ctx.destination);
    for (const t of stream.getAudioTracks()) t.addEventListener("mute", () => { if (mic.st) mic.st.mutes++; });
    let node, kind;
    try {
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" })));
      node = new AudioWorkletNode(ctx, "hs-cap"); node.port.onmessage = (e) => pushPcm(e.data); kind = "worklet";
    } catch {
      node = ctx.createScriptProcessor(4096, 1, 1); node.onaudioprocess = (e) => pushPcm(e.inputBuffer.getChannelData(0).slice(0)); kind = "script";
    }
    src.connect(node); node.connect(sink);
    Object.assign(mic, { ctx, stream, src, node, sink, kind });
    if (ctx.state !== "running") { try { await ctx.resume(); } catch { /* stats */ } }
    return false;
  }
  function releaseCapture() {
    clearTimeout(mic.idle); mic.idle = null;
    for (const f of [() => mic.node.disconnect(), () => mic.src.disconnect(), () => mic.stream.getTracks().forEach((t) => t.stop()),
      () => mic.ctx.state !== "closed" && mic.ctx.close()]) { try { f(); } catch { /* gone */ } }
    Object.assign(mic, { ctx: null, stream: null, src: null, node: null, sink: null });
  }
  function micDebug(why) {
    const st = mic.st || newStats(), tr = mic.stream && mic.stream.getAudioTracks()[0];
    return { why, held_s: +((performance.now() - st.t0) / 1000).toFixed(2), frames: st.frames, in_s: +st.inS.toFixed(2),
      sent_s: +(st.sent / 16000).toFixed(2), node: mic.kind, reused: st.reused, ctx: mic.ctx ? mic.ctx.state : "none",
      rate: mic.ctx ? mic.ctx.sampleRate : 0, track: tr ? tr.readyState : "none", muted: tr ? tr.muted : false, mutes: st.mutes, ws_buf: ws ? ws.bufferedAmount : -1 };
  }
  async function startMic(open = false) {
    if (!navigator.mediaDevices?.getUserMedia) { toast("No microphone here. Type your question instead."); return; }
    if (!wsOpen) { toast("Onset is still starting. Try again in a moment."); return; }
    clearTimeout(mic.idle); mic.idle = null;
    let reused;
    try { reused = await openCapture(); }
    catch { releaseCapture(); toast("Onset needs microphone access. Allow it in System Settings › Privacy › Microphone."); return; }
    Object.assign(mic, { on: true, buf: [], acc: 0, cnt: 0, frac: 0, ratio: mic.ctx.sampleRate / 16000, st: newStats() });
    mic.st.reused = reused ? 1 : 0;
    if (typed.open) { box.value = ""; typed.open = false; }
    send({ t: "start", modes: MODES, open: !!open });
    startRun("mic");
    cur = null;
    post({ t: "mic", on: true });
    paintMic();
  }
  function pushPcm(x) {
    if (!mic.on) return;
    mic.st.frames++; mic.st.inS += x.length / (mic.ctx ? mic.ctx.sampleRate : 48000);
    let sq = 0;
    for (let i = 0; i < x.length; i++) {
      const s = x[i]; sq += s * s;
      mic.acc += s; mic.cnt++; mic.frac += 1;
      if (mic.frac >= mic.ratio) { mic.buf.push(mic.acc / mic.cnt); mic.acc = 0; mic.cnt = 0; mic.frac -= mic.ratio; }
    }
    mic.level = 0.8 * mic.level + 0.2 * Math.sqrt(sq / Math.max(1, x.length));
    while (mic.buf.length >= 1600) {
      const f = new Float32Array(mic.buf.splice(0, 1600));
      if (ws && ws.readyState === 1) { ws.send(f.buffer); mic.st.sent += f.length; }
    }
    $("#micRing").style.transform = `scale(${1 + Math.min(0.35, mic.level * 6)})`;
  }
  function stopMic(sendStop = true, why = "release") {
    if (!mic.on) return;
    mic.on = false;
    if (mic.buf.length && ws && ws.readyState === 1) { mic.st.sent += mic.buf.length; ws.send(new Float32Array(mic.buf.splice(0)).buffer); }
    if (sendStop) send({ t: "stop", dbg: micDebug(why) });
    post({ t: "mic", on: false });
    mic.idle = setTimeout(releaseCapture, MIC_KEEP_MS);
    $("#micRing").style.transform = "";
    paintMic();
  }
  function paintMic() {
    $("#micBtn").classList.toggle("live", mic.on);
    $("#micBtn").setAttribute("aria-label", mic.on ? "Stop" : "Talk");
    $("#hint").innerHTML = mic.on ? "Listening… release <kbd>Space</kbd> or click to stop" : "Hold <kbd>Space</kbd> to talk · <kbd>/</kbd> to type";
  }
  $("#micBtn").addEventListener("click", () => (mic.on ? stopMic(true, "click") : startMic(true)));

  // hold Space to talk (when not typing); "/" or ⌘K types; Esc stops
  const ptt = { held: false, tail: null };
  const typing = () => document.activeElement === box;
  addEventListener("keydown", (e) => {
    if (e.code === "Space" && !typing() && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      if (e.repeat || ptt.held) return;
      ptt.held = true;
      if (ptt.tail) { clearTimeout(ptt.tail); ptt.tail = null; if (mic.on) return; }
      if (!mic.on) startMic(false);
      return;
    }
    if ((e.key === "/" && !typing()) || (e.key === "k" && e.metaKey)) { e.preventDefault(); box.focus(); box.select(); return; }
    if (e.key === "Escape") { if (mic.on) stopMic(true, "escape"); box.blur(); return; }
    if (e.key === "n" && e.metaKey) { e.preventDefault(); newChat(); }
  });
  addEventListener("keyup", (e) => {
    if (e.code !== "Space" || !ptt.held) return;
    ptt.held = false;
    // a short tail: the last word is still arriving when the key comes up
    if (mic.on) ptt.tail = setTimeout(() => { ptt.tail = null; stopMic(true, "release"); }, 350);
  });
  addEventListener("blur", () => { if (ptt.held) { ptt.held = false; if (mic.on) ptt.tail = setTimeout(() => { ptt.tail = null; stopMic(true, "window blur"); }, 500); } });

  // ------------------------------------------------------------------ new chat, settings, replay (automation)
  function newChat() {
    if (mic.on) stopMic(true, "new chat");
    thread.querySelectorAll(".turn").forEach((n) => n.remove());
    turns.clear(); cur = null; fresh = true;
    $("#hello").hidden = false; $("#newChat").hidden = true;
    box.focus();
  }
  $("#newChat").addEventListener("click", newChat);
  $("#settingsBtn").addEventListener("click", () => post({ t: "settings" }));
  const showSettings = () => { $("#settingsBtn").hidden = !bridge(); paintNotice(); };
  showSettings();
  addEventListener("pywebviewready", showSettings);      // Windows: the api is injected after the page loads
  function replay(file) {
    if (!file) return;
    if (mic.on) stopMic(true, "replay");
    if (!send({ t: "replay", file, modes: MODES })) return;
    startRun("replay");
    cur = null;
  }

  connect();
  pollHealth();
  box.focus();
  window.__hs = { get run() { return run; }, get ready() { return wsOpen && warm; }, replay,
    mic: { start: (open) => startMic(!!open), stop: () => stopMic(true, "automation"), stats: () => micDebug("peek"), get on() { return mic.on; } } };
})();
