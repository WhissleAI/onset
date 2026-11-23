/* Memory graph layout helpers, no DOM: shared by the small panel (web/graph.js) and the full-size view
   (web/graphbig.js). tests/test_graph_layout.py runs them under node.
   - augment(graph): the server graph + a "you" node for what the user told us (home, facts)
   - rank(): which nodes get a label first (the focus, the current turn, recency, degree)
   - placeLabels(): greedy collision-aware labels (below / above / right / left of the node, else none)
   - fit(): scale + centre positions into a box (fit-to-bounds)
   - force(): a small force layout; groupLayout(): one cell per turn, packed in rows, labels kept apart */
(function (root) {
  "use strict";

  const ellipsize = (s, max) => {
    s = String(s ?? "");
    return s.length > max ? s.slice(0, Math.max(1, max - 1)).trimEnd() + "…" : s;
  };
  const turnsOf = (n) => (n && Array.isArray(n.turns) ? n.turns.filter((t) => typeof t === "number") : []);
  const firstTurn = (n) => { const t = turnsOf(n); return t.length ? Math.min(...t) : 0; };
  const lastTurn = (n) => { const t = turnsOf(n); return t.length ? Math.max(...t) : 0; };
  const titleCase = (s) => String(s).replace(/\b\w/g, (c) => c.toUpperCase());

  function augment(graph) {
    const g = graph || {};
    const nodes = (g.nodes || []).map((n) => ({ ...n }));
    const user = g.user || [];
    const ids = new Set(nodes.map((n) => n.id));
    if (user.length) { nodes.push({ id: "__you", label: "you", kind: "you", user: true }); ids.add("__you"); }
    const edges = (g.edges || []).filter((e) => ids.has(e.a) && ids.has(e.b)).map((e) => ({ ...e }));
    for (const u of user) {
      if (u.key === "home") {
        const k = String(u.v).toLowerCase();
        if (!ids.has(k)) { nodes.push({ id: k, label: titleCase(u.v), kind: "place", user: true }); ids.add(k); }
        edges.push({ a: "__you", b: k, rel: "lives in", kind: "user" });
      } else {
        const k = "__u_" + u.key;
        if (!ids.has(k)) { nodes.push({ id: k, label: String(u.v), kind: "fact", user: true, said: u.label }); ids.add(k); }
        edges.push({ a: "__you", b: k, rel: "", kind: "user" });
      }
    }
    const turns = (g.turns || []).map((t) => t.t).filter((t) => typeof t === "number");
    const now = turns.length ? Math.max(...turns) : Math.max(0, ...nodes.map(lastTurn));
    return { nodes, edges, now };
  }

  function degree(edges) {
    const d = {};
    for (const e of edges) { d[e.a] = (d[e.a] || 0) + 1; d[e.b] = (d[e.b] || 0) + 1; }
    return d;
  }

  // label priority: the entity a follow-up resolved to, then the current turn's, then recency and connections
  function rank(nodes, edges, focus, now) {
    const deg = degree(edges);
    return nodes.map((n, i) => ({
      n, i,
      s: (n.id === focus ? 1e6 : 0) + (now && turnsOf(n).includes(now) ? 1e4 : 0) + (n.kind === "you" ? 600 : 0)
        + lastTurn(n) * 100 + Math.min(8, deg[n.id] || 0) * 20 + (n.qid ? 5 : 0),
    })).sort((a, b) => b.s - a.s || a.i - b.i).map((x) => x.n);
  }

  const overlap = (a, b, m) => {
    m = m || 0;
    return a.x < b.x + b.w + m && b.x < a.x + a.w + m && a.y < b.y + b.h + m && b.y < a.y + a.h + m;
  };
  const inside = (box, B) => !B || (box.x >= B.x && box.y >= B.y && box.x + box.w <= B.x + B.w && box.y + box.h <= B.y + B.h);

  // items [{id, x, y, w, h, r, cands?}] in priority order. A label is kept only where it overlaps no placed label,
  // no node but its own and stays in bounds; max caps how many are shown. Returns {id: {x, y, w, h, anchor}}.
  function placeLabels(items, opts) {
    const o = opts || {};
    const max = o.max == null ? Infinity : o.max, gap = o.gap == null ? 2 : o.gap;
    const placed = (o.placed || []).slice(), obstacles = o.obstacles || [], out = {};
    let n = 0;
    for (const it of items) {
      if (n >= max) break;
      const r = it.r || 0;
      const cands = it.cands || [
        { x: it.x - it.w / 2, y: it.y + r + gap, anchor: "below" },
        { x: it.x - it.w / 2, y: it.y - r - gap - it.h, anchor: "above" },
        { x: it.x + r + gap + 1, y: it.y - it.h / 2, anchor: "right" },
        { x: it.x - r - gap - 1 - it.w, y: it.y - it.h / 2, anchor: "left" },
      ];
      for (const c of cands) {
        const box = { x: c.x, y: c.y, w: it.w, h: it.h };
        const B = o.bounds;
        if (B && (c.anchor === "below" || c.anchor === "above" || c.anchor === "mid")) {
          // slide a centred label sideways to stay inside, as long as it still covers its node
          const nx = Math.max(B.x, Math.min(B.x + B.w - box.w, box.x));
          if (it.x == null || (nx <= it.x && nx + box.w >= it.x)) box.x = nx;
        }
        if (!inside(box, B)) continue;
        if (placed.some((p) => overlap(p, box, 1))) continue;
        if (obstacles.some((ob) => ob.id !== it.id && overlap(ob, box))) continue;
        placed.push(box);
        out[it.id] = { ...box, anchor: c.anchor };
        n++;
        break;
      }
    }
    return out;
  }

  // scale + centre points into [pad.l, W - pad.r] x [pad.t, H - pad.b]; never magnify more than maxScale
  function fit(points, W, H, pad, maxScale) {
    const P = pad || { l: 10, r: 10, t: 10, b: 10 };
    if (!points.length) return { s: 1, map: (p) => ({ x: W / 2, y: H / 2 }) };
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of points) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
    const bw = Math.max(1e-6, x1 - x0), bh = Math.max(1e-6, y1 - y0);
    const aw = Math.max(1, W - P.l - P.r), ah = Math.max(1, H - P.t - P.b);
    const s = Math.min(aw / bw, ah / bh, maxScale || 1.5);
    const ox = P.l + (aw - bw * s) / 2 - x0 * s, oy = P.t + (ah - bh * s) / 2 - y0 * s;
    return { s, map: (p) => ({ x: ox + p.x * s, y: oy + p.y * s }) };
  }

  // a small force layout around (0, 0); pos {id: {x, y}} is kept between calls so the graph grows stably.
  // aniso > 1 spreads nodes sideways (labels are wider than tall).
  function force(nodes, edges, pos, opt) {
    const o = { iters: 160, len: 62, rep: 1600, gx: 0.012, gy: 0.02, aniso: 1, rnd: Math.random, ...(opt || {}) };
    const ids = new Set(nodes.map((n) => n.id));
    for (const n of nodes) {
      if (pos[n.id]) continue;
      const nb = edges.find((e) => (e.a === n.id && pos[e.b]) || (e.b === n.id && pos[e.a]));
      const p = nb ? pos[nb.a === n.id ? nb.b : nb.a] : { x: 0, y: 0 };
      const a = o.rnd() * Math.PI * 2;
      pos[n.id] = { x: p.x + Math.cos(a) * o.len * 0.6, y: p.y + Math.sin(a) * o.len * 0.45 };
    }
    for (const k of Object.keys(pos)) if (!ids.has(k)) delete pos[k];
    const P = nodes.map((n) => pos[n.id]);
    const idx = Object.fromEntries(nodes.map((n, i) => [n.id, i]));
    const E = edges.filter((e) => idx[e.a] != null && idx[e.b] != null && e.a !== e.b);
    for (let it = 0; it < o.iters; it++) {
      const t = 1 - it / o.iters;
      const F = P.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
        let dx = (P[i].x - P[j].x) / o.aniso, dy = P[i].y - P[j].y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) { dx = o.rnd() - 0.5; dy = o.rnd() - 0.5; d2 = 1; }
        const d = Math.sqrt(d2), f = o.rep / d2;
        F[i].x += (dx / d) * f; F[i].y += (dy / d) * f;
        F[j].x -= (dx / d) * f; F[j].y -= (dy / d) * f;
      }
      for (const e of E) {
        const i = idx[e.a], j = idx[e.b];
        const dx = P[j].x - P[i].x, dy = P[j].y - P[i].y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1, f = (d - o.len) * 0.06;
        F[i].x += (dx / d) * f; F[i].y += (dy / d) * f;
        F[j].x -= (dx / d) * f; F[j].y -= (dy / d) * f;
      }
      const m = 6 * t + 0.5;
      for (let i = 0; i < P.length; i++) {
        F[i].x -= P[i].x * o.gx; F[i].y -= P[i].y * o.gy;
        P[i].x += Math.max(-m, Math.min(m, F[i].x));
        P[i].y += Math.max(-m, Math.min(m, F[i].y));
      }
    }
    return pos;
  }

  // push overlapping boxes apart ({id, x, y, w, h} centred on x, y); a few passes are enough for small groups
  function separate(boxes, passes, margin) {
    const m = margin == null ? 6 : margin;
    for (let k = 0; k < (passes || 60); k++) {
      let moved = false;
      for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        const ox = (a.w + b.w) / 2 + m - Math.abs(a.x - b.x), oy = (a.h + b.h) / 2 + m - Math.abs(a.y - b.y);
        if (ox <= 0 || oy <= 0) continue;
        moved = true;
        if (ox / (a.w + b.w) < oy / (a.h + b.h)) {
          const s = (a.x < b.x || (a.x === b.x && i < j) ? -1 : 1) * ox / 2;
          a.x += s; b.x -= s;
        } else {
          const s = (a.y < b.y || (a.y === b.y && i < j) ? -1 : 1) * oy / 2;
          a.y += s; b.y -= s;
        }
      }
      if (!moved) break;
    }
    return boxes;
  }

  // full-size view: one cell per turn (an entity sits in the turn it was first heard; what you told us in a "you"
  // cell), each laid out on its own so its labels never collide, cells packed left to right in rows of maxW.
  // measure(label) -> px width at the label font. Returns {pos, cells, width, height}.
  function groupLayout(graph, opt) {
    const o = { maxW: 960, minCellW: 190, labelMax: 28, fontH: 16, pad: 16, head: 40, gap: 18, measure: (s) => s.length * 7, rnd: Math.random, ...(opt || {}) };
    const { nodes, edges, now } = augment(graph);
    const groups = new Map();
    for (const n of nodes) {
      const g = n.user ? "you" : (firstTurn(n) || 0);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(n);
    }
    // a turn that added nothing new ("when was it founded?") still gets a cell: it went back to earlier entities
    for (const t of (graph && graph.turns) || []) if (typeof t.t === "number" && !groups.has(t.t)) groups.set(t.t, []);
    const keys = [...groups.keys()].sort((a, b) => (a === "you" ? -1 : b === "you" ? 1 : a - b));
    const refsOf = (t) => (t === "you" ? [] : nodes.filter((n) => turnsOf(n).includes(t) && firstTurn(n) < t).map((n) => n.id));
    const cells = [], pos = {};
    for (const key of keys) {
      const ns = groups.get(key);
      const refs = refsOf(key);
      if (!ns.length) {
        cells.push({ key, now: key === now, w: o.minCellW, h: o.head + 30, nodes: [], refs, place: () => {} });
        continue;
      }
      const inG = new Set(ns.map((n) => n.id));
      const local = {};
      ns.forEach((n, i) => {
        const a = (i / Math.max(1, ns.length)) * Math.PI * 2;
        local[n.id] = { x: Math.cos(a) * 40 * Math.min(3, ns.length / 2), y: Math.sin(a) * 30 * Math.min(3, ns.length / 2) };
      });
      force(ns, edges.filter((e) => inG.has(e.a) && inG.has(e.b)), local, { iters: 200, len: 80, rep: 2600, aniso: 2, gx: 0.02, gy: 0.03, rnd: o.rnd });
      const boxes = ns.map((n) => {
        const w = Math.max(24, o.measure(ellipsize(n.label, o.labelMax))) + 10;
        return { id: n.id, x: local[n.id].x, y: local[n.id].y, w, h: 18 + o.fontH };
      });
      separate(boxes, 120, 8);
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const b of boxes) { x0 = Math.min(x0, b.x - b.w / 2); x1 = Math.max(x1, b.x + b.w / 2); y0 = Math.min(y0, b.y - b.h / 2); y1 = Math.max(y1, b.y + b.h / 2); }
      const w = Math.max(o.minCellW, x1 - x0 + 2 * o.pad), h = y1 - y0 + o.pad + o.head + (refs.length ? 18 : 0);
      const dx = (w - (x1 - x0)) / 2 - x0;                     // centre the group in its cell
      cells.push({ key, now: key === now, w, h, nodes: ns.map((n) => n.id), refs,
                   // the node sits at the top of its box, its label under it
                   place: (cx, cy) => { for (const b of boxes) pos[b.id] = { x: cx + dx + b.x, y: cy + o.head + (b.y - b.h / 2 - y0) + 9 }; } });
    }
    // pack rows
    let x = 0, y = 0, rowH = 0, width = 0;
    for (const c of cells) {
      if (x > 0 && x + c.w > o.maxW) { x = 0; y += rowH + o.gap; rowH = 0; }
      c.x = x; c.y = y;
      c.place(x, y);
      x += c.w + o.gap; rowH = Math.max(rowH, c.h);
      width = Math.max(width, c.x + c.w);
      delete c.place;
    }
    // rows: stretch each cell to its row's height so the grid reads as columns per turn
    const rows = {};
    for (const c of cells) rows[c.y] = Math.max(rows[c.y] || 0, c.h);
    for (const c of cells) c.h = rows[c.y];
    return { pos, cells, width, height: y + rowH, now };
  }

  const api = { ellipsize, firstTurn, lastTurn, augment, rank, placeLabels, overlap, fit, force, separate, groupLayout };
  root.HSGraphLayout = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
