/* Onset demo transport (GitHub Pages: no Python server).
   The page is the real web UI (web/, copied at build time). This file loads first and stands in for the server:
   - window.WebSocket -> DemoSocket: "replay" / "start" / "type" from web/app.js pick a recorded run
     (data/runs/<id>.json, captured by scripts/pages/record.py from the real server) and its events are delivered on
     their recorded clock, anchored to the audio that is playing, so words, chips and answers land as they did live.
   - fetch("/api/...") and fetch("/audio/...") -> static copies under data/; the <audio id="player"> src likewise.
   - the mic is off (no server to hear it); typing is scripted for the typed scenario.
   Nothing here changes what the UI does with an event: it is the same app.js, fed the same events. */
(() => {
  "use strict";
  const DATA = "data/";
  const NativeWS = window.WebSocket;
  const nativeFetch = window.fetch.bind(window);
  const idx = { loaded: null, data: null };
  const load = () => (idx.loaded ||= nativeFetch(DATA + "index.json").then((r) => r.json()).then((d) => (idx.data = d)));
  const runCache = new Map();
  const getRun = (id) => {
    if (!runCache.has(id)) runCache.set(id, nativeFetch(DATA + "runs/" + id + ".json").then((r) => r.json()));
    return runCache.get(id);
  };
  const audioUrl = (name) => DATA + "audio/" + String(name).replace(/\.wav$/i, "") + ".mp3";

  // ------------------------------------------------------------------ fetch: /api/* and /audio/* from data/
  const recentAudio = [];          // conversation turns are fetched (decoded) right before "start": that names the run
  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    if (path.startsWith("/api/")) {
      const name = path.slice(5).split("?")[0];
      const d = await load();
      if (name === "questions") return json(d.questions);
      if (name === "conversations") return json(d.conversations);
      if (name === "health") return json(d.health);
      return nativeFetch(DATA + "api/" + name + ".json").then((r) => (r.ok ? r : json({})));
    }
    if (path.startsWith("/audio/")) {
      const name = decodeURIComponent(path.slice(7));
      const d = await load();
      if (!d.audio.includes(name)) return new Response("not found", { status: 404 });
      recentAudio.push({ name, at: performance.now() });
      if (recentAudio.length > 8) recentAudio.shift();
      return nativeFetch(audioUrl(name));
    }
    return nativeFetch(input, init);
  };
  const json = (o) => new Response(JSON.stringify(o), { status: 200, headers: { "Content-Type": "application/json" } });

  // the player's src: "/audio/x.wav" -> data/audio/x.mp3
  const player = document.getElementById("player");
  if (player) {
    const desc = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "src");
    Object.defineProperty(player, "src", {
      configurable: true,
      get() { return desc.get.call(this); },
      set(v) {
        const m = /^\/audio\/(.+)$/.exec(String(v));
        desc.set.call(this, m ? audioUrl(decodeURIComponent(m[1])) : v);
      },
    });
    player.addEventListener("playing", () => { if (sock.cur && sock.cur.waitAudio && sock.cur.t0 == null) sock.cur.anchor(); });
  }

  // ------------------------------------------------------------------ the socket
  const sock = { cur: null, inst: null };
  const syncLog = [];
  class DemoSocket {
    constructor(url) {
      this.url = url; this.readyState = 0; this.binaryType = "arraybuffer";
      this.onopen = this.onclose = this.onerror = this.onmessage = null;
      sock.inst = this;
      load().then((d) => {
        this.readyState = 1;
        if (this.onopen) this.onopen({});
        this._emit({ ...d.ready });
      }).catch((err) => { console.warn("demo data", err); this.readyState = 3; });
    }
    _emit(e) { if (this.readyState === 1 && this.onmessage) this.onmessage({ data: JSON.stringify(e) }); }
    close() { this.readyState = 3; stopRun(); }
    send(msg) {
      if (typeof msg !== "string") return;                 // audio frames: the recorded run already heard them
      let m;
      try { m = JSON.parse(msg); } catch { return; }
      const d = idx.data;
      if (!d) return;
      if (m.t === "hello") { for (const e of d.hello) this._emit({ ...e }); return; }
      if (m.t === "forget") { this._emit({ t: "memory", mode: "sys", forgot: true, graph: d.emptyGraph }); return; }
      if (m.t === "stop" || m.t === "type" || m.t === "arc_reset" || m.t.startsWith("persona")) return;
      if (m.t === "replay" || m.t === "start") {
        const sc = pick(m, d);
        if (!sc) {
          stopRun();
          this._emit({ t: "error", mode: "sys", error: "This page plays recorded runs. Pick a scenario, or download the Mac app to ask your own." });
          setTimeout(() => this._emit({ t: "run_done", mode: "sys", rid: m.rid }), 50);
          return;
        }
        playRun(this, sc, m);
      }
    }
  }
  DemoSocket.CONNECTING = 0; DemoSocket.OPEN = 1; DemoSocket.CLOSING = 2; DemoSocket.CLOSED = 3;
  window.WebSocket = function (url, p) {
    return /\/ws$/.test(String(url)) ? new DemoSocket(url) : new NativeWS(url, p);
  };
  Object.assign(window.WebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });

  function pick(m, d) {
    if (m.t === "replay") return d.scenarios.find((s) => s.file === m.file);
    if (m.typed) return d.scenarios.find((s) => s.typed);
    const recent = recentAudio.filter((a) => performance.now() - a.at < 8000).map((a) => a.name);
    return d.scenarios.find((s) => Array.isArray(s.files) && s.files.every((f) => recent.includes(f)));
  }

  function stopRun() {
    const c = sock.cur;
    if (!c) return;
    clearInterval(c.tick); clearTimeout(c.fallback);
    sock.cur = null;
  }

  // deliver one recorded run: events at their recorded offset from the run's start, the clock anchored to the audio
  async function playRun(ws, sc, m) {
    stopRun();
    const modes = new Set(Array.isArray(m.modes) && m.modes.length ? m.modes : ["headstart", "baseline"]);
    const c = { sc, rid: m.rid, i: 0, t0: null, sent: performance.now(), waitAudio: !sc.typed, events: null, tick: null, fallback: null };
    c.anchor = () => {
      // first audio frame playing: the recorded clock starts at the start message, the audio a beat later (decode,
      // output latency); anchor so recorded t = audio time
      const lag = player && !player.paused ? player.currentTime * 1000 : 0;
      c.t0 = performance.now() - lag + 60;      // + the audio output path (measured ~60 ms in Chromium)
    };
    sock.cur = c;
    if (!c.waitAudio) c.t0 = c.sent;
    else c.fallback = setTimeout(() => { if (c.t0 == null) c.t0 = c.sent; }, 1500);   // autoplay blocked: run anyway
    const rec = await getRun(sc.id);
    if (sock.cur !== c) return;
    c.events = rec.events.filter((x) => {
      const md = x.e.mode;
      return md === "sys" || md === "ear" || md == null || modes.has(md);
    });
    if (player && c.waitAudio && !player.paused && player.currentTime > 0 && c.t0 == null) c.anchor();
    c.tick = setInterval(() => {
      if (sock.cur !== c || c.t0 == null) return;
      const now = performance.now() - c.t0;
      while (c.i < c.events.length && c.events[c.i].at <= now) {
        const x = c.events[c.i++], e = x.e;
        if (e.t === "words" && player && !player.paused && syncLog.length < 400)   // drift check (tests): recorded vs audio clock
          syncLog.push({ sc: sc.id, at: x.at, audio: Math.round(player.currentTime * 1000), late: Math.round(now - x.at) });
        ws._emit(m.rid != null ? { ...e, rid: m.rid } : e);
      }
      if (c.i >= c.events.length) stopRun();
    }, 15);
  }

  // ------------------------------------------------------------------ page: banner, scenario chips, no mic
  addEventListener("keydown", (e) => {                     // hold-Space-to-talk is off on the static page
    if (e.code === "Space" && !(e.target && e.target.closest && e.target.closest("input, textarea, button"))) {
      e.stopImmediatePropagation(); e.preventDefault();
      if (!e.repeat) hint();
    }
  }, true);
  let typing = false;
  addEventListener("submit", (e) => {
    if (e.target && e.target.id === "typeForm" && !typing) { e.preventDefault(); e.stopImmediatePropagation(); hint(); }
  }, true);
  function hint() {
    const b = document.getElementById("demoBar");
    if (!b) return;
    b.classList.remove("pulse"); void b.offsetWidth; b.classList.add("pulse");
  }

  async function typeScenario(sc) {
    const box = document.getElementById("typeBox"), form = document.getElementById("typeForm");
    if (!box || !form || typing) return;
    typing = true;
    box.value = "";
    try {
      for (let i = 1; i <= sc.text.length; i++) {
        await new Promise((r) => setTimeout(r, sc.char_ms || 110));
        box.value = sc.text.slice(0, i);
        box.dispatchEvent(new Event("input", { bubbles: true }));
      }
      await new Promise((r) => setTimeout(r, sc.enter_ms || 500));
      form.requestSubmit();
    } finally { typing = false; }
  }

  function playScenario(sc) {
    const app = window.__hs;
    if (!app) return;
    if (sc.typed) return typeScenario(sc);
    const box = document.getElementById("typeBox");
    if (box && !typing) box.value = "";
    if (sc.files) {
      const tab = document.querySelector('#qtabs button[data-set="conv"]');
      if (tab && !tab.classList.contains("on")) tab.click();
      const b = document.querySelector(`#qlist button[data-file="${CSS.escape(sc.id)}"]`);
      if (b) b.click();
      return;
    }
    const tab = document.querySelector('#qtabs button[data-set="web"]');
    if (tab && !tab.classList.contains("on")) tab.click();
    const b = document.querySelector(`#qlist button[data-file="${CSS.escape(sc.file)}"]`);
    if (b) b.click(); else app.replay(sc.file);
  }

  function bar(d) {
    const el = document.createElement("div");
    el.id = "demoBar"; el.className = "demo-bar";
    el.innerHTML = `<div class="db-row"><a class="db-home" href="../">← Onset</a>
      <p class="db-msg"><b>Recorded runs of the real app.</b> <span>Download the Mac app to try it live.</span></p>
      <a class="db-dl" href="${d.download}">Download for Mac</a></div>
      <div class="db-row db-pick"><span class="db-lab">Pick a scenario</span><div class="db-chips" role="list"></div></div>
      <p class="db-note" id="demoNote"></p>`;
    const chips = el.querySelector(".db-chips");
    for (const sc of d.scenarios) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "db-chip"; b.setAttribute("role", "listitem"); b.dataset.sc = sc.id;
      b.innerHTML = `<small>${esc(sc.tag)}</small>${esc(sc.title)}`;
      b.title = sc.blurb || "";
      b.addEventListener("click", () => {
        chips.querySelectorAll(".on").forEach((n) => n.classList.remove("on"));
        b.classList.add("on");
        el.querySelector("#demoNote").textContent = sc.blurb || "";
        playScenario(sc);
      });
      chips.appendChild(b);
    }
    document.body.insertBefore(el, document.body.firstChild);
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const boot = () => load().then((d) => {
    bar(d);
    const box = document.getElementById("typeBox");
    const ph = document.querySelector("#transcript .placeholder");
    if (ph) ph.textContent = "Pick a scenario above. Each one replays a recorded run of the app, with its audio (sound on).";
    if (box) { box.readOnly = true; box.placeholder = "Typed questions: pick “Search while you type” above"; }
    // warm the browser cache with every clip, so a click plays at once (the clips are small)
    for (const a of d.audio) nativeFetch(audioUrl(a)).catch(() => {});
    const auto = new URLSearchParams(location.search).get("play");
    if (auto) { const sc = d.scenarios.find((s) => s.id === auto); if (sc) setTimeout(() => playScenario(sc), 600); }
  });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
  window.__demo = { syncLog, get data() { return idx.data; }, play: (id) => playScenario(idx.data.scenarios.find((s) => s.id === id)) };
})();
