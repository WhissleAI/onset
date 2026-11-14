/* Results view: the "Results" button in the top bar opens a full-window slide. Not the "Sources" toggle (that one
   switches the race between answers and search results).

   Data: GET /api/scorecard (server/scorecard.json: the organizers' six problems, one measured number each, with the doc,
   section and date it was measured in, shown as a footnote + tooltip) and this session's meter (web/meter.js
   HSMeter.state(): questions, paid searches, est. $, $ per 100 questions and median time to answer, Onset vs plain
   search). Every number is measured; $ is an estimate and says so. "Save as image" draws the same slide on a canvas
   (1920x1080 PNG, current theme): a download in a browser, a save panel in the Mac app (window.webkit bridge "onset").
   Keyboard: Esc closes, Tab stays inside, focus returns to the button. Works in WKWebView (no external libraries). */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  let data = null, root = null, lastFocus = null, timer = null, loading = null;

  // ------------------------------------------------------------------ data
  async function load() {
    if (data) return data;
    if (!loading) loading = fetch("/api/scorecard", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    data = await loading;
    loading = null;
    return data;
  }
  const srcTip = (s) => (s ? `${s.doc} · ${s.section}${s.date ? " · measured " + s.date : ""}${s.note ? "\n" + s.note : ""}` : "");

  function median(a) {
    const v = a.filter((x) => x != null && isFinite(x)).sort((x, y) => x - y);
    if (!v.length) return null;
    const m = v.length >> 1;
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }
  function fmtUsd(v) {
    if (v == null || !isFinite(v)) return "—";
    if (v === 0) return "$0";
    if (v >= 1) return "$" + v.toFixed(2);
    if (v >= 0.01) return "$" + v.toFixed(3);
    return "$" + v.toFixed(4);
  }
  const fmtS = (ms) => (ms == null || !isFinite(ms) ? "—" : (ms / 1000).toFixed(2) + " s");

  // this session, from the cost meter (questions each mode answered)
  function live() {
    const M = window.HSMeter && window.HSMeter.state ? window.HSMeter.state() : null;
    if (!M) return null;
    const hs = M.T.headstart, bl = M.T.baseline;
    if (!hs && !bl) return null;
    const per100 = (t) => (t && t.n ? (t.usd / t.n) * 100 : null);
    const be = M.prices && M.prices.web_backend;
    return {
      n: hs ? hs.n : 0, nBl: bl ? bl.n : 0, hs, bl,
      paid: hs ? hs.paid : null, paidBl: bl ? bl.paid : null,
      usd: hs ? hs.usd : null, usdBl: bl ? bl.usd : null,
      p100: per100(hs), p100Bl: per100(bl),
      med: median(M.rows.map((r) => r.hsReady)), medBl: median(M.rows.map((r) => r.blReady)),
      note: "$ is an estimate from list prices" + (be === "telem" ? "; the web lane here is Telem, each engine at its own list price" :
        be && be !== "tavily" ? `; the web lane here is ${be}, priced as Tavily searches` : ""),
    };
  }

  // ------------------------------------------------------------------ render
  function ensureRoot() {
    if (root) return root;
    root = document.createElement("div");
    root.id = "scorecard";
    root.className = "sc";
    root.hidden = true;
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-labelledby", "scTitle");
    document.body.appendChild(root);
    root.addEventListener("click", (e) => {
      if (e.target === root) return close();
      const b = e.target.closest("button[data-sc]");
      if (!b) return;
      if (b.dataset.sc === "close") close();
      else if (b.dataset.sc === "png") savePng(b);
    });
    return root;
  }

  function liveHtml() {
    const L = live();
    if (!L || (!L.n && !L.nBl)) {
      return `<p class="sc-empty">Ask a question or replay one, then come back: this fills in with what this session measured.</p>`;
    }
    const raced = L.nBl > 0 && L.n > 0;
    const vs = (txt) => `<small>${raced ? "plain search " + txt : L.n ? "plain search: run Race to compare" : ""}</small>`;
    const tile = (label, big, sub) => `<div class="sc-lt"><label>${label}</label><b>${big}</b>${sub}</div>`;
    return `<div class="sc-ltiles">` +
      tile("questions answered", String(L.n || L.nBl), raced ? `<small>by both, side by side</small>` : `<small>${L.n ? "Onset" : "plain search only"}</small>`) +
      tile("paid searches", L.paid != null ? String(L.paid) : "—", vs(L.paidBl != null ? String(L.paidBl) : "—")) +
      tile("spent (est.)", fmtUsd(L.usd), vs(fmtUsd(L.usdBl))) +
      tile("per 100 questions (est.)", fmtUsd(L.p100), vs(fmtUsd(L.p100Bl))) +
      tile("median time to answer", fmtS(L.med), vs(fmtS(L.medBl))) +
      `</div><p class="sc-fine">After you stop talking, to the first word of the answer. ${esc(L.note)}.</p>`;
  }

  function render() {
    const el = ensureRoot();
    const d = data;
    if (!d) {
      el.innerHTML = `<div class="sc-panel"><header class="sc-head"><div><h2 id="scTitle">Results</h2></div>${actions(false)}</header>
        <p class="sc-empty">The results could not be loaded from this server. Try again in a moment.</p></div>`;
      return;
    }
    const refs = [];
    const ref = (s) => {
      if (!s) return "";
      let i = refs.findIndex((r) => r.doc === s.doc && r.section === s.section);
      if (i < 0) { refs.push(s); i = refs.length - 1; }
      return `<sup class="sc-ref" title="${esc(srcTip(s))}">${i + 1}</sup>`;
    };
    const hero = (d.headline || []).map((h) => `<div class="sc-hero-t"><b>${esc(h.big)}${ref(h.source)}</b><span>${esc(h.label)}</span><small>${esc(h.detail)}</small></div>`).join("");
    const cards = (d.problems || []).map((p) => `
      <article class="sc-card" title="${esc(srcTip(p.source))}">
        <div class="sc-pn"><span>${esc(p.n)}</span>${esc(p.problem)}</div>
        <div class="sc-big">${esc(p.big)}${ref(p.source)}</div>
        <div class="sc-unit">${esc(p.unit)}</div>
        <div class="sc-vs">${esc(p.vs)}</div>
        <p class="sc-ans">${esc(p.answer)}</p>
      </article>`).join("");
    const now = d.now || {};
    const nowTxt = [now.cost_mode ? `budget: ${now.cost_mode}` : "", now.web_backend ? `web: ${now.web_backend}` : "", now.keystone ? "Keystone on" : ""].filter(Boolean).join(" · ");
    el.innerHTML = `<div class="sc-panel">
      <header class="sc-head">
        <div><h2 id="scTitle">${esc(d.title || "Results")}</h2><p class="sc-sub">${esc(d.subtitle || "")}</p></div>
        ${actions(true)}
      </header>
      <section class="sc-hero" aria-label="Headline numbers">${hero}</section>
      <section class="sc-grid" aria-label="The six problems">${cards}</section>
      <section class="sc-live" aria-label="This session">
        <h3>This session <span>${esc(nowTxt)}</span></h3>
        <div id="scLive">${liveHtml()}</div>
      </section>
      <footer class="sc-foot">
        <p>${esc(d.lane_note || "")}</p>
        <ol>${refs.map((s) => `<li>${esc(s.doc)} · ${esc(s.section)}${s.date ? ` · ${esc(s.date)}` : ""}</li>`).join("")}</ol>
        ${modelsLine(d.models)}
      </footer>
    </div>`;
  }
  // "Model versions · last trained <date>" (server/scorecard.json "models", written by the training loop after a
  // promoted cycle, quoted from its PROGRESS.md section like every other number here)
  function modelsLine(m) {
    if (!m || !m.line) return "";
    const by = Object.values(m.by_model || {}).filter((x) => x && x.version)
      .map((x) => `${esc(x.name)} ${esc(x.version)}${x.trained ? ` (${esc(x.trained)})` : ""}`).join(" · ");
    const src = m.source ? `${m.source.doc} · ${m.source.section}` : "";
    return `<p class="sc-models" title="${esc(src)}">${esc(m.line)}${by ? `<span>${by}</span>` : ""}</p>`;
  }
  function actions(png) {
    return `<div class="sc-actions">${png ? `<button type="button" class="sc-btn" data-sc="png">Save as image</button>` : ""}` +
      `<button type="button" class="sc-btn sc-x" data-sc="close" aria-label="Close results (Esc)">Close <kbd>Esc</kbd></button></div>`;
  }

  // ------------------------------------------------------------------ open / close + keyboard
  async function open() {
    lastFocus = document.activeElement;
    ensureRoot();
    if (!data) { render(); root.hidden = false; await load(); }
    render();
    root.hidden = false;
    document.body.classList.add("sc-on");
    const c = root.querySelector('[data-sc="close"]');
    if (c) c.focus();
    clearInterval(timer);
    timer = setInterval(() => { const n = $("#scLive"); if (n && !root.hidden) n.innerHTML = liveHtml(); }, 1000);
  }
  function close() {
    if (!root || root.hidden) return;
    root.hidden = true;
    document.body.classList.remove("sc-on");
    clearInterval(timer);
    timer = null;
    const b = $("#scOpen");
    (lastFocus && document.contains(lastFocus) ? lastFocus : b)?.focus?.();
  }
  const isOpen = () => !!root && !root.hidden;
  // capture phase on window: while open, the page's own keys (Space = push-to-talk, 1-9 replays, D details) never fire
  function onKey(e) {
    if (!isOpen()) return;
    if (e.type === "keydown" && e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); return; }
    if (e.type === "keydown" && e.key === "Tab") {
      const f = Array.from(root.querySelectorAll("button, [href], [tabindex]:not([tabindex='-1'])")).filter((x) => !x.disabled && x.offsetParent !== null);
      if (f.length) {
        const first = f[0], last = f[f.length - 1];
        if (e.shiftKey && (document.activeElement === first || !root.contains(document.activeElement))) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && (document.activeElement === last || !root.contains(document.activeElement))) { e.preventDefault(); first.focus(); }
      }
    }
    e.stopPropagation();
  }
  addEventListener("keydown", onKey, true);
  addEventListener("keyup", onKey, true);

  // ------------------------------------------------------------------ the slide as an image (1920x1080, current theme)
  function wrap(ctx, text, maxW) {
    const words = String(text || "").split(/\s+/), lines = [];
    let cur = "";
    for (const w of words) {
      const t = cur ? cur + " " + w : w;
      if (ctx.measureText(t).width <= maxW || !cur) cur = t; else { lines.push(cur); cur = w; }
    }
    if (cur) lines.push(cur);
    return lines;
  }
  function rrect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  async function drawSlide() {
    const d = data || await load();
    if (!d) throw new Error("no results");
    try { await document.fonts.ready; } catch { /* older engines */ }
    const cs = getComputedStyle(document.documentElement);
    const C = (k, f) => (cs.getPropertyValue(k) || "").trim() || f;
    const col = { bg: C("--bg", "#f4f5f8"), card: C("--card", "#fff"), ink: C("--ink", "#0f1424"), ink2: C("--ink-2", "#4a5268"),
      ink3: C("--ink-3", "#8a91a5"), line: C("--line", "#e3e6ee"), hs: C("--hs", "#0e9f6e") };
    const W = 1920, H = 1080, P = 80, F = 'Inter, -apple-system, "Segoe UI", sans-serif';
    const cv = document.createElement("canvas");
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = col.bg; ctx.fillRect(0, 0, W, H);
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = col.ink; ctx.font = `800 54px ${F}`; ctx.fillText(d.title || "Results", P, 116);
    ctx.fillStyle = col.ink2; ctx.font = `500 24px ${F}`; ctx.fillText(d.subtitle || "", P, 156);
    // headline tiles
    const hs = d.headline || [], gap = 28, hw = (W - 2 * P - gap * (hs.length - 1)) / Math.max(1, hs.length);
    hs.forEach((h, i) => {
      const x = P + i * (hw + gap), y = 192;
      ctx.fillStyle = col.card; rrect(ctx, x, y, hw, 150, 18); ctx.fill(); ctx.strokeStyle = col.line; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = col.hs; ctx.font = `800 54px ${F}`; ctx.fillText(h.big, x + 28, y + 70);
      ctx.fillStyle = col.ink; ctx.font = `600 22px ${F}`; ctx.fillText(h.label, x + 28, y + 104);
      ctx.fillStyle = col.ink3; ctx.font = `400 18px ${F}`; ctx.fillText(wrap(ctx, h.detail, hw - 56)[0] || "", x + 28, y + 132);
    });
    // six problem cards
    const ps = d.problems || [], cw = (W - 2 * P - gap * 2) / 3, ch = 290;
    ps.forEach((p, i) => {
      const x = P + (i % 3) * (cw + gap), y = 376 + Math.floor(i / 3) * (ch + gap);
      ctx.fillStyle = col.card; rrect(ctx, x, y, cw, ch, 18); ctx.fill(); ctx.strokeStyle = col.line; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = col.ink3; ctx.font = `700 18px ${F}`; ctx.fillText(`${p.n}  ${String(p.problem).toUpperCase()}`, x + 28, y + 44);
      ctx.fillStyle = col.ink; ctx.font = `800 76px ${F}`; ctx.fillText(p.big, x + 28, y + 128);
      ctx.fillStyle = col.ink2; ctx.font = `600 22px ${F}`; ctx.fillText(p.unit, x + 28, y + 164);
      ctx.fillStyle = col.hs; ctx.font = `600 19px ${F}`; ctx.fillText(wrap(ctx, p.vs, cw - 56)[0] || "", x + 28, y + 196);
      ctx.fillStyle = col.ink2; ctx.font = `400 19px ${F}`;
      wrap(ctx, p.answer, cw - 56).slice(0, 2).forEach((ln, k) => ctx.fillText(ln, x + 28, y + 236 + k * 26));
    });
    ctx.fillStyle = col.ink3; ctx.font = `400 16px ${F}`;
    const docs = [...new Set([...hs, ...ps].map((x) => x.source && x.source.doc).filter(Boolean))];
    ctx.fillText(wrap(ctx, d.lane_note || "", W - 2 * P)[0] || "", P, H - 52);
    ctx.fillText("Sources: " + docs.join(", ") + " · github.com/WhissleAI/headstart", P, H - 26);
    return cv;
  }
  async function savePng(btn) {
    const old = btn.textContent;
    btn.disabled = true; btn.textContent = "Saving…";
    try {
      const cv = await drawSlide();
      const url = cv.toDataURL("image/png");
      const nb = window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.onset;
      if (nb) nb.postMessage({ t: "savePNG", name: "Onset results.png", data: url });
      else {
        const a = document.createElement("a");
        a.href = url; a.download = "onset-results.png";
        document.body.appendChild(a); a.click(); a.remove();
      }
      btn.textContent = "Saved";
    } catch (err) {
      console.error("scorecard png", err);
      btn.textContent = "Could not save";
    }
    setTimeout(() => { btn.textContent = old; btn.disabled = false; }, 1600);
  }

  // ------------------------------------------------------------------ wire up
  function boot() {
    const b = $("#scOpen");
    if (b) b.addEventListener("click", () => (isOpen() ? close() : open()));
    if (/[#&]results\b/.test(location.hash)) open();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
  window.HSScorecard = { open, close, isOpen, drawSlide, load };
})();
