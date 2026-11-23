/* Memory, full size: the session's knowledge graph (web/graph.js) in a window-sized view.
   One soft cell per turn (an entity sits in the turn it was first heard; what you told Onset in a "you" cell), the
   current turn outlined, its entities drawn larger. Every node is labelled; relation labels show where they fit and
   always for the entity you point at. Zoom (wheel, pinch, + / − / Fit, keys + - 0), pan (drag the background), drag
   nodes, hover or click an entity for its type, turns, relations and source; search dims everything else.
   Same overlay rules as the Results view (web/scorecard.js): Esc closes, Tab stays inside, focus returns to what
   opened it, and no key reaches the page while it is open (Space is not push-to-talk here).
   Plain SVG, no libraries; works in WKWebView. graph.js calls HSGraphBig.open(graph, focus, from) and .update(). */
(function () {
  "use strict";
  const GL = window.HSGraphLayout;
  const NS = "http://www.w3.org/2000/svg";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const KIND = { person: "person", place: "place", org: "organization", thing: "thing", you: "you", fact: "something you told me" };
  const SRC = (e) => (e.kind === "wikidata" ? "Wikidata fact" : e.kind === "said" ? "from an answer" : e.kind === "user" ? "you told me"
    : e.rel === "mentioned" ? "mentioned in an answer" : "asked together");
  const RELTXT = (e) => e.rel || (e.kind === "co" ? "asked with" : e.kind === "user" ? "said" : "related");
  const FONT = 12.5, LAB_MAX = 30;

  let root = null, svg = null, world = null, stage = null, side = null, search = null;
  let G = { nodes: [], edges: [], user: [], turns: [] }, focusId = null, from = null;
  let L = null, sig = "", view = { k: 1, x: 0, y: 0 }, sel = null, hover = null, query = "", dirty = true;
  let A = { nodes: [], edges: [], now: 0 }, byId = {};

  let mctx = null;
  function measure(t, font) {
    try {
      if (!mctx) mctx = document.createElement("canvas").getContext("2d");
      mctx.font = font || `500 ${FONT}px ${getComputedStyle(document.body).getPropertyValue("--sans") || "sans-serif"}`;
      return mctx.measureText(t).width;
    } catch { return String(t).length * 7; }
  }
  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, String(v));
    if (parent) parent.appendChild(n);
    return n;
  }

  // ------------------------------------------------------------------ DOM
  function ensure() {
    if (root) return root;
    root = document.createElement("div");
    root.className = "mg";
    root.id = "memBig";
    root.hidden = true;
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-labelledby", "mgTitle");
    root.innerHTML = `<div class="mg-panel">
      <header class="mg-head">
        <div class="mg-title"><h2 id="mgTitle">Memory</h2><p class="mg-sub" id="mgSub"></p></div>
        <div class="mg-tools">
          <input id="mgSearch" class="mg-search" type="search" placeholder="Find in memory" aria-label="Find an entity in memory" spellcheck="false" autocomplete="off">
          <div class="mg-zoom" role="group" aria-label="Zoom">
            <button type="button" class="mg-btn" data-mg="out" aria-label="Zoom out" title="Zoom out (−)">−</button>
            <button type="button" class="mg-btn" data-mg="fit" aria-label="Fit to window" title="Fit to window (0)">Fit</button>
            <button type="button" class="mg-btn" data-mg="in" aria-label="Zoom in" title="Zoom in (+)">+</button>
          </div>
          <button type="button" class="mg-btn mg-x" data-mg="close" aria-label="Close memory (Esc)">Close <kbd>Esc</kbd></button>
        </div>
      </header>
      <div class="mg-body">
        <div class="mg-stage" id="mgStage">
          <svg class="mg-svg" id="mgSvg" tabindex="0" role="img" aria-label="Knowledge graph of this conversation, one group per turn"></svg>
          <p class="mg-hint">Drag to move · scroll or pinch to zoom · click an entity for details</p>
        </div>
        <aside class="mg-side" aria-label="Details">
          <div class="mg-detail" id="mgDetail" aria-live="polite"></div>
          <div class="mg-legend" aria-label="Legend">
            <h4>Legend</h4>
            <ul>
              <li><i class="mg-dot k-person"></i>person</li><li><i class="mg-dot k-place"></i>place</li>
              <li><i class="mg-dot k-org"></i>organization</li><li><i class="mg-dot k-thing"></i>thing</li>
              <li><i class="mg-dot k-you"></i>you · what you told me</li>
            </ul>
            <ul class="mg-lines">
              <li><svg viewBox="0 0 26 6" aria-hidden="true"><line class="mg-e k-wikidata" x1="1" y1="3" x2="25" y2="3"/></svg>Wikidata fact</li>
              <li><svg viewBox="0 0 26 6" aria-hidden="true"><line class="mg-e k-said" x1="1" y1="3" x2="25" y2="3"/></svg>from an answer</li>
              <li><svg viewBox="0 0 26 6" aria-hidden="true"><line class="mg-e k-co" x1="1" y1="3" x2="25" y2="3"/></svg>asked or mentioned together</li>
              <li><svg viewBox="0 0 26 6" aria-hidden="true"><line class="mg-e k-user" x1="1" y1="3" x2="25" y2="3"/></svg>you told me</li>
              <li><i class="mg-cellkey"></i>current turn</li>
            </ul>
          </div>
        </aside>
      </div>
    </div>`;
    document.body.appendChild(root);
    svg = root.querySelector("#mgSvg");
    stage = root.querySelector("#mgStage");
    side = root.querySelector("#mgDetail");
    search = root.querySelector("#mgSearch");
    world = null;
    root.addEventListener("click", (e) => {
      const b = e.target.closest("[data-mg]");
      if (b) {
        const a = b.dataset.mg;
        if (a === "close") close();
        else if (a === "in") zoomBy(1.25);
        else if (a === "out") zoomBy(0.8);
        else if (a === "fit") fitView(true);
        return;
      }
      const nb = e.target.closest("[data-node]");
      if (nb && side.contains(nb)) { select(nb.dataset.node, true); return; }
      const tb = e.target.closest("[data-turn]");
      if (tb && side.contains(tb)) { showTurn(tb.dataset.turn); }
    });
    search.addEventListener("input", () => { query = search.value.trim().toLowerCase(); draw(); });
    wireStage();
    return root;
  }

  // ------------------------------------------------------------------ data + layout
  function relayout(force) {
    A = GL.augment(G);
    byId = Object.fromEntries(A.nodes.map((n) => [n.id, n]));
    const s = A.nodes.map((n) => n.id + ":" + (n.turns || []).join(".")).join("|") + "#" + A.edges.length;
    if (!force && s === sig && L) return false;
    sig = s;
    const W = Math.max(480, (stage && stage.clientWidth) || 900);
    L = GL.groupLayout(G, { maxW: W - 32, measure: (t) => measure(t), labelMax: LAB_MAX, fontH: FONT + 4 });
    if (sel && !byId[sel]) sel = null;
    return true;
  }
  const turnQ = (t) => ((G.turns || []).find((x) => x.t === t) || {}).q || "";

  // ------------------------------------------------------------------ view (zoom + pan)
  function applyView() { if (world) world.setAttribute("transform", `translate(${view.x.toFixed(1)},${view.y.toFixed(1)}) scale(${view.k.toFixed(4)})`); }
  function fitView(anim) {
    if (!L || !stage) return;
    const sw = stage.clientWidth, sh = stage.clientHeight, pad = 20;
    const k = Math.max(0.35, Math.min(1.5, (sw - 2 * pad) / Math.max(1, L.width), (sh - 2 * pad) / Math.max(1, L.height)));
    view = { k, x: (sw - L.width * k) / 2, y: Math.max(pad, (sh - L.height * k) / 2) };
    if (anim) smooth(); else applyView();
  }
  function smooth() {
    if (!world) return;
    world.classList.add("mg-anim");
    applyView();
    setTimeout(() => world && world.classList.remove("mg-anim"), 220);
  }
  function zoomAt(f, cx, cy) {
    const k = Math.max(0.25, Math.min(4, view.k * f));
    const r = k / view.k;
    view = { k, x: cx - (cx - view.x) * r, y: cy - (cy - view.y) * r };
    applyView();
  }
  function zoomBy(f) { zoomAt(f, stage.clientWidth / 2, stage.clientHeight / 2); }
  function showTurn(t) {
    const c = L && L.cells.find((x) => String(x.key) === String(t));
    if (!c) return;
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const k = Math.max(0.5, Math.min(1.6, (sw - 60) / c.w, (sh - 60) / c.h));
    view = { k, x: sw / 2 - (c.x + c.w / 2) * k, y: sh / 2 - (c.y + c.h / 2) * k };
    smooth();
  }

  function wireStage() {
    const pt = (e) => { const r = svg.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    let drag = null, gesturing = false, g0 = 1;
    svg.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      const ng = e.target.closest("[data-id]");
      const p = pt(e);
      drag = { id: ng ? ng.dataset.id : null, x: p.x, y: p.y, vx: view.x, vy: view.y, moved: false, pid: e.pointerId };
      if (drag.id) { const q = L.pos[drag.id]; drag.nx = q.x; drag.ny = q.y; }
      try { svg.setPointerCapture(e.pointerId); } catch { /* old engines */ }
      svg.classList.add(drag.id ? "mg-dragnode" : "mg-panning");
    });
    svg.addEventListener("pointermove", (e) => {
      if (!drag) {
        const ng = e.target.closest("[data-id]");
        const h = ng ? ng.dataset.id : null;
        if (h !== hover) { hover = h; draw(); }
        return;
      }
      const p = pt(e), dx = p.x - drag.x, dy = p.y - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 3) return;
      drag.moved = true;
      if (drag.id) { L.pos[drag.id] = { x: drag.nx + dx / view.k, y: drag.ny + dy / view.k }; draw(); }
      else { view.x = drag.vx + dx; view.y = drag.vy + dy; applyView(); }
    });
    const end = (e) => {
      if (!drag) return;
      const d = drag;
      drag = null;
      svg.classList.remove("mg-dragnode", "mg-panning");
      try { svg.releasePointerCapture(e.pointerId); } catch { /* fine */ }
      if (!d.moved && e.type === "pointerup") select(d.id && d.id === sel ? null : d.id, false);
    };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
    svg.addEventListener("pointerleave", () => { if (!drag && hover) { hover = null; draw(); } });
    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (gesturing) return;
      const p = pt(e);
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      zoomAt(Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0018)), p.x, p.y);
    }, { passive: false });
    // Safari / WKWebView trackpad pinch
    svg.addEventListener("gesturestart", (e) => { e.preventDefault(); gesturing = true; g0 = 1; });
    svg.addEventListener("gesturechange", (e) => {
      e.preventDefault();
      const r = svg.getBoundingClientRect();
      zoomAt(e.scale / g0, (e.clientX || r.left + r.width / 2) - r.left, (e.clientY || r.top + r.height / 2) - r.top);
      g0 = e.scale;
    });
    svg.addEventListener("gestureend", (e) => { e.preventDefault(); gesturing = false; });
    if (window.ResizeObserver) {
      let w0 = 0;
      new ResizeObserver(() => {
        if (!isOpen()) return;
        const w = stage.clientWidth;
        if (Math.abs(w - w0) > 40) { w0 = w; relayout(true); draw(); fitView(false); }
      }).observe(stage);
    }
  }

  // ------------------------------------------------------------------ drawing
  function neighbours(id) {
    const s = new Set([id]);
    for (const e of A.edges) { if (e.a === id) s.add(e.b); if (e.b === id) s.add(e.a); }
    return s;
  }
  function draw() {
    if (!root || root.hidden || !L) return;
    const ae = document.activeElement;
    const keep = ae && svg.contains(ae) && ae.getAttribute ? ae.getAttribute("data-id") : null;   // keyboard focus survives a redraw
    svg.innerHTML = "";
    world = el("g", { class: "mg-world" }, svg);
    applyView();
    const P = L.pos, now = A.now;
    const nt = (G.turns || []).length, ne = A.edges.filter((e) => e.kind !== "co").length;
    root.querySelector("#mgSub").textContent = `${(G.nodes || []).length} entities · ${ne} relation${ne === 1 ? "" : "s"} · ${nt} turn${nt === 1 ? "" : "s"} · this conversation`;
    const act = sel || hover;
    const near = act ? neighbours(act) : null;
    const match = query ? new Set(A.nodes.filter((n) => (n.label + " " + (n.desc || "") + " " + (KIND[n.kind] || "")).toLowerCase().includes(query)).map((n) => n.id)) : null;
    const lit = (id) => (near ? near.has(id) : match ? match.has(id) : true);
    const font = `500 ${FONT}px ${getComputedStyle(document.body).getPropertyValue("--sans") || "sans-serif"}`;
    const rOf = (n) => (n.kind === "you" ? 9 : now && (n.turns || []).includes(now) ? 9 : 7);

    // turn cells
    const gC = el("g", { class: "mg-cells" }, world);
    const headBoxes = [];
    for (const c of L.cells) {
      const g = el("g", { class: `mg-cell${c.now ? " now" : ""}${c.key === "you" ? " you" : ""}` }, gC);
      el("rect", { x: c.x, y: c.y, width: c.w, height: c.h, rx: 14, class: "mg-cell-r" }, g);
      const label = c.key === "you" ? "You told me" : c.key ? `Turn ${c.key}` : "Earlier";
      const t = el("text", { x: c.x + 12, y: c.y + 18, class: "mg-cell-t" }, g);
      t.textContent = label + (c.now ? " · now" : "");
      const q = c.key !== "you" ? turnQ(c.key) : "";
      if (q) {
        const room = c.w - 24;
        let s = `“${q}”`;
        const f2 = `400 11px ${getComputedStyle(document.body).getPropertyValue("--sans") || "sans-serif"}`;
        while (s.length > 4 && measure(s, f2) > room) s = s.slice(0, -2);
        if (s !== `“${q}”`) s = s.replace(/[\s”]*$/, "") + "…”";
        const tq = el("text", { x: c.x + 12, y: c.y + 32, class: "mg-cell-q" }, g);
        tq.textContent = s;
        el("title", {}, tq).textContent = q;
      }
      if (c.refs && c.refs.length) {
        const names = c.refs.map((id) => (byId[id] ? byId[id].label : id));
        const f3 = `600 11px ${getComputedStyle(document.body).getPropertyValue("--sans") || "sans-serif"}`;
        let s = "back to " + names.join(", ");
        while (s.length > 12 && measure(s, f3) > c.w - 24) s = s.slice(0, -2);
        if (s !== "back to " + names.join(", ")) s = s.replace(/[\s,]*$/, "") + "…";
        const tr = el("text", { x: c.x + 12, y: c.nodes.length ? c.y + c.h - 9 : c.y + 54, class: "mg-cell-ref" }, g);
        tr.textContent = s;
        el("title", {}, tr).textContent = "this turn went back to " + names.join(", ");
      }
      headBoxes.push({ x: c.x, y: c.y, w: c.w, h: 38 });
    }

    // edges: straight inside a turn, curved across turns
    const cellOf = {};
    for (const c of L.cells) for (const id of c.nodes) cellOf[id] = c.key;
    const gE = el("g", { class: "mg-edges" }, world);
    const mids = [];
    for (const e of A.edges) {
      const a = P[e.a], b = P[e.b];
      if (!a || !b) continue;
      const hot = act && (e.a === act || e.b === act);
      const dim = (act && !hot) || (!act && match && !(match.has(e.a) || match.has(e.b)));
      let d, at;
      if (cellOf[e.a] === cellOf[e.b]) { d = `M${a.x},${a.y}L${b.x},${b.y}`; at = (t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }); }
      else {
        const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, bend = Math.min(60, len * 0.15);
        const cx = (a.x + b.x) / 2 - (dy / len) * bend, cy = (a.y + b.y) / 2 + (dx / len) * bend;
        d = `M${a.x},${a.y}Q${cx},${cy} ${b.x},${b.y}`;
        at = (t) => ({ x: (1 - t) * (1 - t) * a.x + 2 * t * (1 - t) * cx + t * t * b.x, y: (1 - t) * (1 - t) * a.y + 2 * t * (1 - t) * cy + t * t * b.y });
      }
      const { x: mx, y: my } = at(0.5);
      el("path", { d, class: `mg-e k-${e.kind}${hot ? " hot" : ""}${dim ? " dim" : ""}` }, gE);
      if (e.kind !== "co" || hot) mids.push({ e, mx, my, at, hot, dim });
    }

    // node label boxes (always shown) are obstacles for relation labels
    const boxes = {};
    for (const n of A.nodes) {
      const p = P[n.id], t = GL.ellipsize(n.label, LAB_MAX), w = measure(t, font) + 2, r = rOf(n);
      boxes[n.id] = { id: n.id, t, x: p.x - w / 2, y: p.y + r + 3, w, h: FONT + 2 };
    }
    const obstacles = A.nodes.map((n) => ({ id: n.id, x: P[n.id].x - rOf(n), y: P[n.id].y - rOf(n), w: 2 * rOf(n), h: 2 * rOf(n) }));
    const rfont = `500 10.5px ${getComputedStyle(document.body).getPropertyValue("--mono") || "monospace"}`;
    const items = mids.sort((x, y) => (y.hot - x.hot) || ((y.e.kind === "said") - (x.e.kind === "said"))).map((m, i) => {
      const t = GL.ellipsize(RELTXT(m.e), 22), w = measure(t, rfont) + 6;
      m.t = t; m.id = "r" + i;
      const cands = [];
      for (const t of [0.5, 0.38, 0.62, 0.28, 0.72]) {
        const q = m.at(t);
        cands.push({ x: q.x - w / 2, y: q.y - 7, anchor: "mid" }, { x: q.x - w / 2, y: q.y - 17, anchor: "mid" }, { x: q.x - w / 2, y: q.y + 3, anchor: "mid" });
      }
      return { id: m.id, w, h: 14, cands };
    });
    const shown = GL.placeLabels(items, { placed: Object.values(boxes).concat(headBoxes), obstacles });
    const gR = el("g", { class: "mg-rels" }, world);
    for (const m of mids) {
      const b = shown[m.id];
      if (!b && !m.hot) continue;
      const x = b ? b.x + b.w / 2 : m.mx, y = b ? b.y + 10.5 : m.my + 3.5;
      const t = el("text", { x: x.toFixed(1), y: y.toFixed(1), "text-anchor": "middle", class: `mg-rl${m.hot ? " hot" : ""}${m.dim ? " dim" : ""}` }, gR);
      t.textContent = m.t;
      el("title", {}, t).textContent = `${byId[m.e.a]?.label || m.e.a} → ${RELTXT(m.e)} → ${byId[m.e.b]?.label || m.e.b} · ${SRC(m.e)}${m.e.turn ? " · turn " + m.e.turn : ""}`;
    }

    // nodes + labels
    const gN = el("g", { class: "mg-nodes" }, world);
    for (const n of A.nodes) {
      const p = P[n.id], r = rOf(n), cur = now && (n.turns || []).includes(now);
      const cls = ["mg-n", "k-" + n.kind, n.user ? "user" : "", cur ? "now" : "", n.id === focusId ? "focus" : "",
        n.id === sel ? "sel" : "", lit(n.id) ? "" : "dim", match && match.has(n.id) ? "match" : ""].filter(Boolean).join(" ");
      const g = el("g", { class: cls, transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`, "data-id": n.id, tabindex: 0, role: "button",
                         "aria-label": `${n.label}, ${KIND[n.kind] || n.kind}${n.turns && n.turns.length ? ", turn " + n.turns.join(", ") : ""}` }, gN);
      if (n.id === focusId || n.id === sel) el("circle", { r: r + 5, class: "mg-halo" }, g);
      el("circle", { r: r + 8, class: "mg-hit" }, g);
      el("circle", { r, class: "mg-c" }, g);
      const b = boxes[n.id];
      const t = el("text", { x: 0, y: (r + 3 + FONT - 1).toFixed(1), "text-anchor": "middle", class: "mg-l" }, g);
      t.textContent = b.t;
      if (b.t !== n.label) el("title", {}, t).textContent = n.label;
      if (keep === n.id) g.focus();
    }
    detail();
  }

  // ------------------------------------------------------------------ details (side panel)
  function relRow(e) {
    const a = byId[e.a], b = byId[e.b];
    const nm = (n, id) => `<button type="button" class="mg-link" data-node="${esc(id)}">${esc(n ? n.label : id)}</button>`;
    return `<li>${nm(a, e.a)} <span class="mg-rel">${esc(RELTXT(e))}</span> → ${nm(b, e.b)}<small>${esc(SRC(e))}${e.turn ? ` · turn ${e.turn}` : ""}</small></li>`;
  }
  function detail() {
    const id = sel || hover;
    const n = id && byId[id];
    if (!n) {
      const turns = (G.turns || []).slice().sort((a, b) => a.t - b.t);
      const cellIds = Object.fromEntries((L ? L.cells : []).map((c) => [String(c.key), c.nodes]));
      side.innerHTML = `<h3 class="mg-d-h">The conversation</h3>` + (turns.length ? `<ol class="mg-turns">${turns.map((t) => {
        const ids = cellIds[String(t.t)] || [];
        const chips = ids.slice(0, 6).map((i) => `<span>${esc(byId[i] ? byId[i].label : i)}</span>`).join("") + (ids.length > 6 ? `<span>+${ids.length - 6}</span>` : "");
        return `<li${t.t === A.now ? ' class="now"' : ""}><button type="button" class="mg-turn" data-turn="${t.t}"><b>${t.t === A.now ? "Now" : "Turn " + t.t}</b>${esc(t.q)}</button>${chips ? `<div class="mg-chips">${chips}</div>` : ""}</li>`;
      }).join("")}</ol>` : `<p class="mg-muted">No questions yet.</p>`) + `<p class="mg-muted">Click a turn to zoom to it, or an entity for its details.</p>`;
      return;
    }
    const ts = (n.turns || []).slice().sort((a, b) => a - b);
    const rels = A.edges.filter((e) => e.a === n.id || e.b === n.id);
    const facts = n.facts ? Object.entries(n.facts) : [];
    const src = n.kind === "you" ? "you, from what you said" : n.user ? "you told me" : n.qid ? `Wikidata ${n.qid}` : "heard in the conversation";
    side.innerHTML = `
      <div class="mg-d-kind"><i class="mg-dot k-${esc(n.kind)}"></i>${esc(KIND[n.kind] || n.kind)}${n.id === focusId ? ' <em>last follow-up meant this</em>' : ""}</div>
      <h3 class="mg-d-h">${esc(n.label)}</h3>
      ${n.desc ? `<p class="mg-d-desc">${esc(n.desc)}</p>` : ""}
      <dl class="mg-dl">
        ${ts.length ? `<dt>Turn</dt><dd>${ts.length === 1 ? "first heard in turn " + ts[0] : `first heard in turn ${ts[0]}; again in ${ts.slice(1).join(", ")}`}</dd>` : ""}
        ${ts.length && turnQ(ts[0]) ? `<dt>Asked</dt><dd>“${esc(turnQ(ts[0]))}”</dd>` : ""}
        <dt>Source</dt><dd>${esc(src)}</dd>
        ${facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}
      </dl>
      <h4>Relations <span>${rels.length}</span></h4>
      ${rels.length ? `<ul class="mg-rels-l">${rels.map(relRow).join("")}</ul>` : `<p class="mg-muted">None yet.</p>`}
      ${sel ? `<p class="mg-muted">Click the background or press Esc to clear.</p>` : ""}`;
  }

  function select(id, center) {
    sel = id && byId[id] ? id : null;
    hover = null;
    draw();
    if (sel && center) {
      const p = L.pos[sel];
      view = { ...view, x: stage.clientWidth / 2 - p.x * view.k, y: stage.clientHeight / 2 - p.y * view.k };
      smooth();
    }
  }

  // ------------------------------------------------------------------ open / close + keyboard
  const isOpen = () => !!root && !root.hidden;
  function open(graph, focus, opener) {
    ensure();
    from = opener || document.activeElement;
    G = graph || G; focusId = focus || null;
    sel = null; hover = null; query = ""; search.value = "";
    root.hidden = false;
    document.body.classList.add("mg-on");
    relayout(true);
    draw();
    fitView(false);
    // focus the graph, not Close: Space (push-to-talk habit) must not close the view
    try { svg.focus({ preventScroll: true }); } catch { svg.focus(); }
  }
  function close() {
    if (!isOpen()) return;
    root.hidden = true;
    document.body.classList.remove("mg-on");
    const back = from && document.contains(from) && from.offsetParent !== null ? from : document.getElementById("memStage");
    from = null;
    back?.focus?.();
  }
  function update(graph, focus) {
    G = graph || G; focusId = focus || null;
    if (!isOpen()) return;
    if (relayout(false)) { draw(); fitView(true); } else draw();
  }
  function onKey(e) {
    if (!isOpen()) return;
    const inInput = e.target === search;
    if (e.type === "keydown") {
      if (e.key === "Escape") {
        e.preventDefault();
        if (inInput && search.value) { search.value = ""; query = ""; draw(); }
        else if (sel) { sel = null; draw(); }
        else close();
      } else if (e.key === "Tab") {
        const f = Array.from(root.querySelectorAll("button, input, [href], [tabindex]:not([tabindex='-1'])")).filter((x) => !x.disabled && (x.offsetParent !== null || x instanceof SVGElement));
        if (f.length) {
          const first = f[0], last = f[f.length - 1];
          if (e.shiftKey && (document.activeElement === first || !root.contains(document.activeElement))) { e.preventDefault(); last.focus(); }
          else if (!e.shiftKey && (document.activeElement === last || !root.contains(document.activeElement))) { e.preventDefault(); first.focus(); }
        }
      } else if (inInput && e.key === "Enter") {
        e.preventDefault();
        const m = A.nodes.find((n) => n.label.toLowerCase().includes(query));
        if (m) select(m.id, true);
      } else if (!inInput && (e.key === "Enter" || e.key === " " || e.code === "Space")) {
        const ng = document.activeElement && document.activeElement.closest && document.activeElement.closest("[data-id]");
        if (ng && root.contains(ng)) { e.preventDefault(); select(ng.dataset.id, true); }
        else if (e.code === "Space" && !(document.activeElement instanceof HTMLButtonElement)) e.preventDefault();
      } else if (!inInput && (e.key === "+" || e.key === "=")) { e.preventDefault(); zoomBy(1.25); }
      else if (!inInput && (e.key === "-" || e.key === "_")) { e.preventDefault(); zoomBy(0.8); }
      else if (!inInput && e.key === "0") { e.preventDefault(); fitView(true); }
    }
    // nothing reaches the page while this is open: no push-to-talk, no 1-9 replays, no D for details
    e.stopPropagation();
  }
  addEventListener("keydown", onKey, true);
  addEventListener("keyup", onKey, true);

  window.HSGraphBig = { open, close, isOpen, update };
})();
