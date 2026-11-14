/* Memory panel: this session's entities and relations as a small live knowledge graph (server/memory.py).
   Nodes come from the session's entity memory (people, places, organisations, things; Wikidata QID when the facts lane
   resolved one), edges from Wikidata facts (founded by, born in, headquarters, ...), from "who / which ..." answers
   (the said edge) and from co-mention. The entity a follow-up was resolved to lights up. User facts ("you live in
   Denver") hang off a "you" node; they stay on this machine, and "forget" clears them here and on the server.
   SVG + a tiny force layout, no libraries. app.js calls HSGraph.connected(send), .start(kind, modes), .event(e). */
(function () {
  "use strict";
  const KEY = "hs-client", OPEN = "hs-mem-open";
  const W = 300, H = 196;
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
  let persist = false, note = "", noteQ = null, flash = 0;

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
        <b>Memory</b><span class="mem-sub" id="memSub">this conversation</span>
        <span class="mem-sp"></span>
        <button class="mem-btn" id="memForget" title="Forget everything remembered for this browser (conversation and your facts), here and on the server">forget</button>
        <button class="mem-btn mem-x" id="memClose" title="Collapse" aria-label="Collapse memory panel">–</button>
      </div>
      <div class="mem-using" id="memUsing" hidden></div>
      <svg class="mem-svg" id="memSvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Knowledge graph of this conversation"></svg>
      <div class="mem-empty" id="memEmpty">Ask a question, then a follow-up ("when was it founded?"). The people, places and companies you talk about appear here.</div>
      <div class="mem-foot" id="memFoot"></div>
    </div>`;
  document.body.appendChild(dock);
  const $ = (id) => document.getElementById(id);
  let open = store.get(OPEN) !== "0";
  function setOpen(v) { open = v; store.set(OPEN, v ? "1" : "0"); paintDock(); }
  $("memPill").addEventListener("click", () => setOpen(true));
  $("memClose").addEventListener("click", () => setOpen(false));
  $("memForget").addEventListener("click", () => {
    if (send) send({ t: "forget" });
    graph = { nodes: [], edges: [], user: [] }; focus = null; pos = {}; note = "";
    paint();
  });

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

  function paintDock() {
    const n = graph.nodes.length + (graph.user || []).length;
    dock.classList.toggle("open", open);
    dock.classList.toggle("has", n > 0);
    $("memN").textContent = String(n);
  }

  // ------------------------------------------------------------------ layout
  function layout() {
    const nodes = graph.nodes.slice();
    const user = graph.user || [];
    if (user.length) nodes.push({ id: "__you", label: "you", kind: "you" });
    const ids = new Set(nodes.map((n) => n.id));
    const edges = (graph.edges || []).filter((e) => ids.has(e.a) && ids.has(e.b));
    for (const u of user) {
      if (u.key === "home") {
        const k = String(u.v).toLowerCase();
        if (!ids.has(k)) { nodes.push({ id: k, label: u.v.replace(/\b\w/g, (c) => c.toUpperCase()), kind: "place", user: true }); ids.add(k); }
        edges.push({ a: "__you", b: k, rel: "lives in", kind: "user" });
      } else {
        const k = "__u_" + u.key;
        nodes.push({ id: k, label: u.v, kind: "fact", user: true });
        edges.push({ a: "__you", b: k, rel: "", kind: "user" });
      }
    }
    // seed new nodes near a neighbour (stable growth), old ones keep their place
    for (const n of nodes) {
      if (pos[n.id]) continue;
      const nb = edges.find((e) => (e.a === n.id && pos[e.b]) || (e.b === n.id && pos[e.a]));
      const p = nb ? pos[nb.a === n.id ? nb.b : nb.a] : { x: W / 2, y: H / 2 };
      const a = Math.random() * Math.PI * 2;
      pos[n.id] = { x: p.x + Math.cos(a) * 40, y: p.y + Math.sin(a) * 30 };
    }
    for (const k of Object.keys(pos)) if (!ids.has(k)) delete pos[k];
    const P = nodes.map((n) => pos[n.id]);
    const idx = Object.fromEntries(nodes.map((n, i) => [n.id, i]));
    for (let it = 0; it < 160; it++) {
      const t = 1 - it / 160;
      const F = P.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
        let dx = P[i].x - P[j].x, dy = P[i].y - P[j].y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 1; }
        const f = 1600 / d2;
        const d = Math.sqrt(d2);
        F[i].x += (dx / d) * f; F[i].y += (dy / d) * f;
        F[j].x -= (dx / d) * f; F[j].y -= (dy / d) * f;
      }
      for (const e of edges) {
        const i = idx[e.a], j = idx[e.b];
        const dx = P[j].x - P[i].x, dy = P[j].y - P[i].y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - 62) * 0.06;
        F[i].x += (dx / d) * f; F[i].y += (dy / d) * f;
        F[j].x -= (dx / d) * f; F[j].y -= (dy / d) * f;
      }
      for (let i = 0; i < P.length; i++) {
        F[i].x += (W / 2 - P[i].x) * 0.012; F[i].y += (H / 2 - P[i].y) * 0.02;
        const m = 6 * t + 0.5;
        P[i].x += Math.max(-m, Math.min(m, F[i].x));
        P[i].y += Math.max(-m, Math.min(m, F[i].y));
        P[i].x = Math.max(34, Math.min(W - 34, P[i].x));
        P[i].y = Math.max(14, Math.min(H - 16, P[i].y));
      }
    }
    return { nodes, edges };
  }

  function svgEl(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, String(v));
    if (parent) parent.appendChild(n);
    return n;
  }

  function paint() {
    paintDock();
    paintNote();
    const svg = $("memSvg");
    svg.innerHTML = "";
    const empty = !graph.nodes.length && !(graph.user || []).length;
    $("memEmpty").hidden = !empty;
    svg.style.display = empty ? "none" : "";
    const turns = (graph.turns || []).length;
    $("memSub").textContent = empty ? (persist ? "kept on this device" : "this conversation")
      : `${graph.nodes.length} entities · ${turns} turn${turns === 1 ? "" : "s"}${persist ? " · kept on this device" : ""}`;
    const foot = $("memFoot");
    foot.innerHTML = (graph.user || []).map((u) => `<span class="mem-fact" title="Said by you; used only as a query word when a question needs it">${esc(u.label)}</span>`).join("");
    if (empty) return;
    const { nodes, edges } = layout();
    const gE = svgEl("g", { class: "mem-edges" }, svg);
    for (const e of edges) {
      const a = pos[e.a], b = pos[e.b];
      const hot = focus && (e.a === focus || e.b === focus);
      svgEl("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: `mem-e k-${e.kind}${hot ? " hot" : ""}` }, gE);
      if (e.rel && e.kind !== "co") {
        const t = svgEl("text", { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 3, class: "mem-el", "text-anchor": "middle" }, gE);
        t.textContent = e.rel.length > 16 ? e.rel.slice(0, 15) + "…" : e.rel;
      }
    }
    const gN = svgEl("g", { class: "mem-nodes" }, svg);
    for (const n of nodes) {
      const p = pos[n.id];
      const hot = n.id === focus;
      const g = svgEl("g", { class: `mem-n k-${n.kind}${hot ? " hot" : ""}${n.user ? " user" : ""}`, transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})` }, gN);
      if (hot) svgEl("circle", { r: 12, class: "mem-halo" }, g);
      svgEl("circle", { r: n.kind === "you" ? 7 : hot ? 7 : 5.5, class: "mem-c" }, g);
      const t = svgEl("text", { x: 0, y: 17, "text-anchor": "middle", class: "mem-l" }, g);
      const lab = n.label.length > 20 ? n.label.slice(0, 19) + "…" : n.label;
      t.textContent = lab;
      const title = svgEl("title", {}, g);
      const facts = n.facts ? Object.entries(n.facts).map(([k, v]) => `${k}: ${v}`).join("; ") : "";
      title.textContent = [n.label, n.kind, n.qid ? `Wikidata ${n.qid}` : "", n.desc || "", facts, n.turns ? `turns ${n.turns.join(", ")}` : ""].filter(Boolean).join(" · ");
    }
  }

  // ------------------------------------------------------------------ events
  function primaryMode(e) { return e.mode === primary || e.mode === "sys"; }
  function event(e) {
    if (!e || !primaryMode(e)) return;
    if (e.t === "memory") {
      if (e.persist !== undefined) persist = !!e.persist;
      if (e.forgot) { focus = null; note = ""; }
      if (e.graph) { graph = e.graph; if (e.graph.focus && !e.used) focus = e.graph.focus; }
      if (e.used) {
        focus = e.used;
        const ts = (e.turns || []).length ? ` <span class="muted">(turn ${e.turns[0]})</span>` : "";
        const ref = e.ref ? esc(e.ref) : "reference";
        note = `<i>memory</i> ${ref} → <b>${esc(e.label || e.used)}</b>${ts}` + (e.inline ? ` <span class="muted">· asked as “${esc(e.inline)}”</span>` : "");
        noteQ = e.q;
        flash = Date.now();
      }
      if (e.learned && e.learned.length) {
        note = `<i>remembered</i> ${e.learned.map((x) => esc(x.label)).join(" · ")} <span class="muted">· stays on this device</span>`;
        noteQ = e.q;
      }
      paint();
      return;
    }
    if (e.t === "frame") {
      if (noteQ != null && e.q !== noteQ && !(e.memory || e.carried)) { note = ""; noteQ = null; paintNote(); }
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
    paintNote();
  }
  function connected(fn) {
    send = fn;
    send({ t: "hello", client: clientId() });
  }

  paint();
  window.HSGraph = { event, start, connected, clientId, get graph() { return graph; } };
})();
