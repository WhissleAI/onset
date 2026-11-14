/* Memory panel: this session's entities and relations as a small live knowledge graph (server/memory.py).
   Nodes come from the session's entity memory (people, places, organisations, things; Wikidata QID when the facts lane
   resolved one), edges from Wikidata facts (founded by, born in, headquarters, ...), from "who / which ..." answers
   (the said edge) and from co-mention. The entity a follow-up was resolved to lights up. User facts ("you live in
   Denver") hang off a "you" node; they stay on this machine, and "forget" clears them here and on the server.
   SVG + a tiny force layout, no libraries. app.js calls HSGraph.connected(send), .start(kind, modes), .event(e). */
(function () {
  "use strict";
  const KEY = "hs-client", OPEN = "hs-mem-open";
  let W = 300, H = 160;            // the small graph is drawn at the panel's own pixel size (no scaled text)
  const LAB_FONT = 10, LAB_MAX = 18;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  function clientId() {
    let id = store.get(KEY);
    if (!id || !/^[A-Za-z0-9_-]{8,64}$/.test(id)) {
      const rnd = (window.crypto && crypto.getRandomValues) ? Array.from(crypto.getRandomValues(new Uint8Array(12)), (b) => b.toString(16).padStart(2, "0")).join("") : String(Math.random()).slice(2) + Date.now();
      id = "c" + rnd;
      store.set(KEY, id);
    }
    return id;
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const NS = "http://www.w3.org/2000/svg";

  let send = null, primary = "headstart", graph = { nodes: [], edges: [], user: [] }, focus = null, pos = {};
  let persist = false, note = "", noteQ = null;
  let voice = null;               // Arc state across questions (server/arcstate.py "arcstate" events)
  const turnsOf = {};

  // ------------------------------------------------------------------ DOM
  const dock = document.createElement("section");
  dock.className = "mem-dock";
  dock.id = "memDock";
  dock.setAttribute("aria-label", "Session memory");
  dock.innerHTML = `
    <button class="mem-pill" id="memPill" title="Session memory: entities and relations of this conversation">
      <span class="mem-dot"></span><b>Memory</b><span class="mem-n" id="memN">0</span>
    </button>
    <div class="mem-body" id="memBody">
      <div class="mem-head">
        <b>Memory</b>
        <span class="mem-sp"></span>
        <button class="mem-btn" id="memVoiceReset" title="Forget how your voice sounds (who is speaking, your usual pace and pitch). What was said stays in memory." hidden>reset voice</button>
        <button class="mem-btn" id="memForget" title="Forget everything remembered for this browser (conversation, your facts and your voice), here and on the server">forget</button>
        <button class="mem-btn mem-x mem-ico" id="memExpand" title="Open the memory graph full size" aria-label="Open the memory graph full size" aria-haspopup="dialog" hidden><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M7 1h4v4M11 1 7.2 4.8M5 11H1V7M1 11l3.8-3.8"/></svg></button>
        <button class="mem-btn mem-x" id="memClose" title="Collapse" aria-label="Collapse memory panel">–</button>
      </div>
      <div class="mem-sub" id="memSub">this conversation</div>
      <div class="mem-using" id="memUsing" hidden></div>
      <div class="mem-voice" id="memVoice" hidden></div>
      <div class="mem-stage" id="memStage" role="button" tabindex="0" aria-haspopup="dialog" title="Open the memory graph full size"
        aria-label="Knowledge graph of this conversation. Open it full size"><svg class="mem-svg" id="memSvg" viewBox="0 0 ${W} ${H}" aria-hidden="true"></svg><span class="mem-zoomhint" aria-hidden="true">expand</span></div>
      <div class="mem-empty" id="memEmpty">Ask a question, then a follow-up ("when was it founded?"). The people, places and companies you talk about appear here.</div>
      <div class="mem-foot" id="memFoot"></div>
      <div class="pers" id="pers" hidden>
        <div class="pers-head"><b>For you</b><span class="mem-sp"></span>
          <button class="mem-btn mem-x" id="persExport" title="Download what Onset learned about how you ask (JSON; no audio, no transcripts)">export</button>
          <button class="mem-btn" id="persClear" title="Clear what Onset learned about how you ask, here and on the server">clear</button>
        </div>
        <div class="pers-sub" id="persSub"></div>
        <div class="pers-body" id="persBody"></div>
      </div>
    </div>`;
  document.body.appendChild(dock);
  const $ = (id) => document.getElementById(id);
  let open = store.get(OPEN) !== "0";
  function setOpen(v) { open = v; store.set(OPEN, v ? "1" : "0"); paint(); }
  $("memPill").addEventListener("click", () => setOpen(true));
  $("memClose").addEventListener("click", () => setOpen(false));
  $("memVoiceReset").addEventListener("click", () => {
    if (send) send({ t: "arc_reset" });
    voice = null;
    paintVoice();
  });
  $("memForget").addEventListener("click", () => {
    voice = null; paintVoice();
    if (send) send({ t: "forget" });
    graph = { nodes: [], edges: [], user: [] }; focus = null; pos = {}; note = "";
    paint();
  });

  // ------------------------------------------------------------------ persona (server/persona.py, PERSONA=1)
  // "Answered for you": why this answer was shaped for this user (shorter: you asked quickly; Jaguar Cars: you asked
  // about the Ford Mustang earlier). The profile itself sits in this panel with export / clear.
  let persOn = false, profile = null, persWhy = { q: null, style: "", sense: "" };
  $("persClear").addEventListener("click", () => {
    if (send) send({ t: "persona_clear" });
    profile = null; persWhy = { q: null, style: "", sense: "" };
    paintPersona(); paintPersNote();
  });
  $("persExport").addEventListener("click", () => { if (send) send({ t: "persona_export" }); });
  function download(obj) {
    try {
      const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url; a.download = "onset-profile.json";
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) { console.error("export", err); }
  }
  function paintPersona() {
    const box = $("pers");
    box.hidden = !persOn;
    if (!persOn) return;
    const p = profile || {};
    const bits = [];
    if (p.length && p.length !== "normal") bits.push(p.length === "short" ? "short answers" : "detailed answers");
    if (p.expertise) bits.push(p.expertise === "beginner" ? "plain words" : "expert");
    if (p.units) bits.push(p.units === "imperial" ? "miles · feet · °F" : "km · metres · °C");
    $("persSub").textContent = p.turns ? `${p.turns} question${p.turns === 1 ? "" : "s"} · kept on this device` : "nothing learned yet";
    const topics = (p.topics || []).slice(0, 6).map((t) => `<span class="mem-fact" title="${esc(t.cat ? "interest: " + t.cat : "interest")}">${esc(t.label)}</span>`).join("");
    $("persBody").innerHTML = (bits.length ? `<div class="pers-line">${bits.map(esc).join(" · ")}</div>` : "")
      + (p.sense ? `<div class="pers-line muted">lately: ${esc(p.sense)}s</div>` : "")
      + (topics ? `<div class="mem-foot">${topics}</div>` : "")
      + (!p.turns ? `<div class="pers-line muted">Answers adapt to how you ask (quick, unsure, units) and what you asked about before.</div>` : "");
  }
  function persNoteEl() {
    let n = document.getElementById("persNote");
    const after = noteEl();
    if (!n && after) {
      n = document.createElement("div");
      n.id = "persNote";
      n.className = "mem-note pers-note";
      n.hidden = true;
      after.insertAdjacentElement("afterend", n);
    }
    return n;
  }
  function paintPersNote() {
    const n = persNoteEl();
    if (!n) return;
    const why = [persWhy.style, persWhy.sense].filter(Boolean);
    n.hidden = !why.length;
    n.innerHTML = why.length ? `<i>answered for you</i> ${why.map(esc).join(" · ")}` : "";
  }
  function personaEvent(e) {
    if (e.on !== undefined) persOn = !!e.on;
    if (e.profile) { profile = e.profile; persOn = true; }
    if (e.cleared) { profile = null; persWhy = { q: null, style: "", sense: "" }; paintPersNote(); }
    if (e.export) download(e.export);
    if (e.q != null && (e.why !== undefined || e.sense !== undefined)) {
      if (persWhy.q !== e.q) persWhy = { q: e.q, style: "", sense: "" };
      if (e.why !== undefined) persWhy.style = e.why || "";
      if (e.sense !== undefined) persWhy.sense = e.sense || "";
      paintPersNote();
    }
    paintPersona();
  }

  // a one-line note under the transcript: what memory did to this question ("using: you live in Denver")
  function noteEl() {
    let n = document.getElementById("memNote");
    const tr = document.getElementById("transcript");
    if (!n && tr && tr.parentElement) {
      n = document.createElement("div");
      n.id = "memNote";
      n.className = "mem-note";
      n.hidden = true;
      tr.insertAdjacentElement("afterend", n);
    }
    return n;
  }
  function paintNote() {
    const n = noteEl();
    if (n) { n.hidden = !note; n.innerHTML = note; }
    const u = $("memUsing");
    u.hidden = !note;
    u.innerHTML = note;
  }

  // details view: the dock sits in the left column under the replay list (no overlap); minimal view: bottom right
  function place() {
    const side = document.querySelector(".side");
    const inSide = !!(side && side.offsetParent !== null && getComputedStyle(side).display !== "none" && window.innerWidth > 980);
    if (inSide && dock.parentElement !== side) side.appendChild(dock);
    else if (!inSide && dock.parentElement !== document.body) document.body.appendChild(dock);
    dock.classList.toggle("in-side", inSide);
  }
  window.addEventListener("resize", place);
  document.addEventListener("keyup", () => setTimeout(place, 50));
  new MutationObserver(() => place()).observe(document.body, { attributes: true, attributeFilter: ["class"] });

  function paintDock() {
    place();
    const n = graph.nodes.length + (graph.user || []).length;
    dock.classList.toggle("open", open);
    dock.classList.toggle("has", n > 0);
    $("memN").textContent = String(n);
  }

  // ------------------------------------------------------------------ layout (web/graphlayout.js)
  const GL = window.HSGraphLayout;
  let measureCtx = null;
  function measure(text, font) {
    try {
      if (!measureCtx) measureCtx = document.createElement("canvas").getContext("2d");
      measureCtx.font = font;
      return measureCtx.measureText(text).width;
    } catch { return String(text).length * 6; }
  }
  const LABEL_FONT = () => `500 ${LAB_FONT}px ${getComputedStyle(document.body).getPropertyValue("--sans") || "sans-serif"}`;

  function svgEl(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, String(v));
    if (parent) parent.appendChild(n);
    return n;
  }
  function tip(n) {
    const facts = n.facts ? Object.entries(n.facts).map(([k, v]) => `${k}: ${v}`).join("; ") : "";
    return [n.label, n.kind, n.qid ? `Wikidata ${n.qid}` : "", n.desc || "", facts, n.turns && n.turns.length ? `turns ${n.turns.join(", ")}` : ""].filter(Boolean).join(" · ");
  }

  // the small graph: force layout, fit to the panel, then labels only where they fit (top N by priority);
  // every node keeps its full name as a tooltip, and the full-size view shows them all
  function paintSmall(svg) {
    const stage = $("memStage");
    const w = Math.round(stage.clientWidth || 300);
    W = Math.max(200, Math.min(420, w));
    H = Math.round(Math.max(150, Math.min(200, W < 260 ? W * 0.8 : W * 0.58)));
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const { nodes, edges, now } = GL.augment(graph);
    GL.force(nodes, edges, pos, { iters: 160, len: 56, rep: 1500, aniso: 1.6, gx: 0.012, gy: 0.03 });
    const f = GL.fit(nodes.map((n) => pos[n.id]), W, H, { l: 14, r: 14, t: 12, b: 22 }, 1.4);
    const P = Object.fromEntries(nodes.map((n) => [n.id, f.map(pos[n.id])]));
    const rOf = (n) => (n.id === focus || n.kind === "you" ? 7 : now && (n.turns || []).includes(now) ? 6 : 5);
    const font = LABEL_FONT();
    const order = GL.rank(nodes, edges, focus, now);
    const maxLabels = W < 260 ? 6 : W < 340 ? 8 : 10;
    const obstacles = nodes.map((n) => ({ id: n.id, x: P[n.id].x - rOf(n) - 1, y: P[n.id].y - rOf(n) - 1, w: 2 * rOf(n) + 2, h: 2 * rOf(n) + 2 }));
    const texts = {};
    const items = order.map((n) => {
      const t = GL.ellipsize(n.label, LAB_MAX);
      texts[n.id] = t;
      return { id: n.id, x: P[n.id].x, y: P[n.id].y, r: rOf(n), w: measure(t, font) + 2, h: LAB_FONT + 3 };
    });
    // labels may cross other nodes' dots (the halo keeps them legible) but never another label
    const labels = GL.placeLabels(items, { bounds: { x: 1, y: 1, w: W - 2, h: H - 2 }, max: maxLabels });
    // relation labels: the focus's edges first, then the current turn's answers; only where they fit
    const placed = Object.values(labels);
    const relItems = edges.filter((e) => e.rel && e.kind !== "co")
      .map((e) => ({ e, s: (focus && (e.a === focus || e.b === focus) ? 100 : 0) + (e.turn === now ? 10 : 0) + (e.kind === "said" ? 2 : 1) }))
      .sort((a, b) => b.s - a.s).slice(0, 6)
      .map(({ e }) => {
        const a = P[e.a], b = P[e.b], t = GL.ellipsize(e.rel, 16);
        const tw = measure(t, `500 8.5px ${getComputedStyle(document.body).getPropertyValue("--mono") || "monospace"}`) + 2;
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        texts["e:" + e.a + ">" + e.b + ">" + e.rel] = t;
        return { id: "e:" + e.a + ">" + e.b + ">" + e.rel, x: mx, y: my, w: tw, h: 11,
                 cands: [{ x: mx - tw / 2, y: my - 12, anchor: "mid" }, { x: mx - tw / 2, y: my + 1, anchor: "mid" }] };
      });
    const relLabels = GL.placeLabels(relItems, { bounds: { x: 1, y: 1, w: W - 2, h: H - 2 }, obstacles, placed, max: 3 });

    const gE = svgEl("g", { class: "mem-edges" }, svg);
    for (const e of edges) {
      const a = P[e.a], b = P[e.b];
      const hot = focus && (e.a === focus || e.b === focus);
      svgEl("line", { x1: a.x.toFixed(1), y1: a.y.toFixed(1), x2: b.x.toFixed(1), y2: b.y.toFixed(1), class: `mem-e k-${e.kind}${hot ? " hot" : ""}` }, gE);
    }
    for (const [id, bx] of Object.entries(relLabels)) {
      const t = svgEl("text", { x: (bx.x + 1).toFixed(1), y: (bx.y + 8.5).toFixed(1), class: "mem-el" }, gE);
      t.textContent = texts[id];
    }
    const gN = svgEl("g", { class: "mem-nodes" }, svg);
    for (const n of nodes) {
      const p = P[n.id];
      const hot = n.id === focus;
      const cur = now && (n.turns || []).includes(now);
      const g = svgEl("g", { class: `mem-n k-${n.kind}${hot ? " hot" : ""}${n.user ? " user" : ""}${cur ? " now" : ""}`, transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})` }, gN);
      if (hot) svgEl("circle", { r: 12, class: "mem-halo" }, g);
      svgEl("circle", { r: rOf(n), class: "mem-c" }, g);
      svgEl("title", {}, g).textContent = tip(n);
    }
    const gL = svgEl("g", { class: "mem-labels" }, svg);
    for (const n of nodes) {
      const bx = labels[n.id];
      if (!bx) continue;
      const t = svgEl("text", { x: (bx.x + 1).toFixed(1), y: (bx.y + LAB_FONT - 0.5).toFixed(1), class: `mem-l${n.id === focus ? " hot" : ""}` }, gL);
      t.textContent = texts[n.id];
      svgEl("title", {}, t).textContent = tip(n);
    }
    const hidden = nodes.length - Object.keys(labels).length;
    $("memStage").dataset.more = hidden > 0 ? `+${hidden} more` : "";
  }

  function paint() {
    paintDock();
    paintNote();
    const svg = $("memSvg");
    svg.innerHTML = "";
    const empty = !graph.nodes.length && !(graph.user || []).length;
    $("memEmpty").hidden = !empty;
    $("memStage").style.display = empty ? "none" : "";
    $("memExpand").hidden = empty;
    const turns = (graph.turns || []).length;
    $("memSub").textContent = empty ? (persist ? "kept on this device" : "this conversation")
      : `${graph.nodes.length} entities · ${turns} turn${turns === 1 ? "" : "s"}${persist ? " · kept on this device" : ""}`;
    const foot = $("memFoot");
    foot.innerHTML = (graph.user || []).map((u) => `<span class="mem-fact" title="Said by you; used only as a query word when a question needs it">${esc(u.label)}</span>`).join("");
    if (window.HSGraphBig) window.HSGraphBig.update(graph, focus);
    if (empty || !open) return;
    paintSmall(svg);
  }

  // click (or Enter) on the small graph, or the expand button: the full-size view (web/graphbig.js)
  function expand(from) { if (window.HSGraphBig && !(graph.nodes.length === 0 && !(graph.user || []).length)) window.HSGraphBig.open(graph, focus, from); }
  $("memStage").addEventListener("click", (e) => expand(e.currentTarget));
  $("memStage").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " " || e.code === "Space") { e.preventDefault(); e.stopPropagation(); expand(e.currentTarget); }
  });
  $("memStage").addEventListener("keyup", (e) => { if (e.code === "Space") { e.preventDefault(); e.stopPropagation(); } });
  $("memExpand").addEventListener("click", (e) => expand(e.currentTarget));
  // the panel's width changes with the window and the details view: redraw at the new size
  if (window.ResizeObserver) {
    let lastW = 0;
    new ResizeObserver(() => {
      const w = Math.round($("memStage").clientWidth);
      if (w && Math.abs(w - lastW) > 2) { lastW = w; paint(); }
    }).observe($("memStage"));
  }

  function paintVoice() {
    const el = $("memVoice"), btn = $("memVoiceReset");
    if (!el) return;
    const on = !!(voice && voice.on && voice.enrolled);
    btn.hidden = !on;
    if (!on) { el.hidden = true; el.innerHTML = ""; return; }
    const who = voice.who === "other" ? "<b>another voice</b> asked this" : "<b>you</b>";
    const n = voice.turns || 0;
    const notes = (voice.notes || []).map(esc).join(" · ");
    el.innerHTML = `<span class="muted">voice</span> ${who}` +
      (voice.voices > 1 ? ` <span class="muted">· ${voice.voices} voices heard</span>` : "") +
      (notes ? ` · ${notes}` : (n ? ` <span class="muted">· learned from ${n} question${n === 1 ? "" : "s"}</span>` : ""));
    el.hidden = false;
  }

  // ------------------------------------------------------------------ events
  function primaryMode(e) { return e.mode === primary || e.mode === "sys"; }
  function event(e) {
    if (!e || !primaryMode(e)) return;
    if (e.t === "persona") { personaEvent(e); return; }
    if (e.t === "frame" && persWhy.q != null && e.q !== persWhy.q) { persWhy = { q: null, style: "", sense: "" }; paintPersNote(); }
    if (e.t === "memory") {
      if (e.persist !== undefined) persist = !!e.persist;
      if (e.forgot) { focus = null; note = ""; }
      if (e.graph) { graph = e.graph; if (e.graph.focus && !e.used) focus = e.graph.focus; }
      if (e.used) { focus = e.used; turnsOf[e.used] = e.turns || []; }
      if (e.learned && e.learned.length) {
        note = `<i>remembered</i> ${e.learned.map((x) => esc(x.label)).join(" · ")} <span class="muted">· stays on this device</span>`;
        noteQ = e.q;
      }
      paint();
      return;
    }
    if (e.t === "arcstate") {
      voice = e.reset === "all" ? null : e;
      paintVoice();
      return;
    }
    if (e.t === "frame") {
      if (noteQ != null && e.q !== noteQ && !(e.memory || e.carried)) { note = ""; noteQ = null; paintNote(); }
      if (e.carried) {
        // the latest frame says what the reference resolved to (partials may resolve differently on the way)
        const k = String(e.carried).toLowerCase();
        const ts = (turnsOf[k] || []).length ? ` <span class="muted">(turn ${turnsOf[k][0]})</span>` : "";
        const ref = (e.memory && e.memory.ref) || "reference";
        const inl = e.memory && e.memory.inline;
        note = `<i>memory</i> ${esc(ref)} → <b>${esc(k.replace(/\b\w/g, (c) => c.toUpperCase()))}</b>${ts}` + (inl ? ` <span class="muted">· asked as “${esc(inl)}”</span>` : "");
        noteQ = e.q;
        if (graph.nodes.some((n) => n.id === k)) focus = k;
        paint();
        return;
      }
      const u = e.memory && e.memory.user;
      if (u && u.length) {
        note = `<i>using</i> ${u.map(esc).join(" · ")} <span class="muted">· from what you told me</span>`;
        noteQ = e.q;
        paintNote();
      }
    }
  }
  function start(kind, modes) {
    primary = (modes && modes[0]) || "headstart";
    note = ""; noteQ = null;
    persWhy = { q: null, style: "", sense: "" };
    paintNote(); paintPersNote();
  }
  function connected(fn) {
    send = fn;
    send({ t: "hello", client: clientId() });
  }

  paint();
  window.HSGraph = { event, start, connected, clientId, get graph() { return graph; } };
})();
