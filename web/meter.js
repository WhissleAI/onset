/* Cost-and-speed meter: per question, Onset vs plain search (Baseline) side by side, plus running session totals.

   Fed by app.js: HSMeter.start(run) on every run, HSMeter.event(e, run) on every server event. Reads the per-question
   "metric" event (ready_ms, paid_calls, free_calls, wasted_fires, answer-model tokens ans_* / waste_*, est. $ from
   server/meter.py) and the "usage" event the speculator sends when the question closes (the final token tally: drafts
   still streaming when the metric went out, a superseded soft answer). Every number here is measured; $ is an estimate
   (list prices in server/config.py METER_*) and says so. A mode that did not run is left out, never imputed.
   Bar   hero ("0.52 s faster · 5.0× cheaper", session averages over questions both modes answered) · this question's
         HS / BL row · session totals · details.
   More  per-question table, price notes. HSMeter.evalSummary(raw) renders the same terms for an eval A/B file. */
(() => {
  "use strict";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const MODES = ["headstart", "baseline"];
  const NAME = { headstart: "Onset", baseline: "Baseline" };
  const SHORT = { headstart: "HS", baseline: "BL" };
  const TOK_KEYS = ["ans_in", "ans_out", "waste_in", "waste_out", "llm_requests", "waste_streams", "tokens_est", "usd",
    "usd_web", "usd_llm", "usd_llm_waste", "price_verified"];
  const rows = [];                // one per question (app.js question object), across runs of this page
  const orphan = { headstart: { in: 0, out: 0, usd: 0 }, baseline: { in: 0, out: 0, usd: 0 } };   // superseded questions
  let prices = null, open = false, root = null, lastRow = null;

  // ------------------------------------------------------------------ formatting
  const fmtS = (ms) => (ms == null || !isFinite(ms) ? "—" : (ms < 0 ? "−" : "") + (Math.abs(ms) / 1000).toFixed(Math.abs(ms) < 9950 ? 2 : 1) + " s");
  const fmtTok = (n) => (n == null ? "—" : n >= 9950 ? (n / 1000).toFixed(0) + "k" : n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(Math.round(n)));
  function fmtUsd(v) {
    if (v == null || !isFinite(v)) return "—";
    if (v === 0) return "$0";
    if (v >= 1) return "$" + v.toFixed(2);
    if (v >= 0.01) return "$" + v.toFixed(3);
    if (v >= 0.0001) return "$" + v.toFixed(4);
    return "<$0.0001";
  }
  const x = (r) => (r >= 9.95 ? r.toFixed(0) : r.toFixed(1)) + "×";

  // ------------------------------------------------------------------ data
  function rowFor(q, run) {
    let r = rows.find((o) => o.q === q);
    if (!r) { r = { q, run, n: rows.length + 1, m: {} }; rows.push(r); }
    return r;
  }
  // ready after the speaker really stopped (same as app.js readyAdj: the latest last-word end either mode saw)
  function ready(r, mode) {
    const met = r.m[mode];
    if (!met || met.ready_ms == null) return null;
    const q = r.q, se = q.se?.[mode] ?? q.speechEnd;
    if (se == null) return met.ready_ms;
    return se + met.ready_ms - (q.speechEnd ?? se);
  }
  const answered = (met) => met && !met.failed && !met.superseded && !met.skipped && met.ready_ms != null;
  const paidOf = (m) => m.paid_calls ?? m.web_paid ?? m.web_calls ?? 0;
  const wastedOf = (m) => m.wasted_fires ?? Math.max(0, (m.spec_fires || 0) - (m.spec_used || 0));
  const tokIn = (m) => (m.ans_in || 0) + (m.waste_in || 0);
  const tokOut = (m) => (m.ans_out || 0) + (m.waste_out || 0);

  function totals() {
    const T = {};
    for (const mode of MODES) {
      const ms = rows.map((r) => r.m[mode]).filter(answered);
      T[mode] = ms.length ? {
        n: ms.length,
        paid: ms.reduce((a, m) => a + paidOf(m), 0), free: ms.reduce((a, m) => a + (m.free_calls || 0), 0),
        wasted: ms.reduce((a, m) => a + wastedOf(m), 0),
        tin: ms.reduce((a, m) => a + tokIn(m), 0) + orphan[mode].in, tout: ms.reduce((a, m) => a + tokOut(m), 0) + orphan[mode].out,
        win: ms.reduce((a, m) => a + (m.waste_in || 0), 0) + orphan[mode].in,
        usd: ms.reduce((a, m) => a + (m.usd || 0), 0) + orphan[mode].usd,
        cards: ms.filter((m) => m.answer_by === "card").length,
      } : null;
    }
    // paired: questions both modes answered (the only fair comparison)
    const pr = rows.filter((r) => answered(r.m.headstart) && answered(r.m.baseline));
    const P = { n: pr.length, saved: 0, hsUsd: 0, blUsd: 0, hsPaid: 0, blPaid: 0, faster: 0 };
    for (const r of pr) {
      const a = ready(r, "headstart"), b = ready(r, "baseline");
      if (a != null && b != null) { P.saved += b - a; if (a < b) P.faster++; }
      P.hsUsd += r.m.headstart.usd || 0; P.blUsd += r.m.baseline.usd || 0;
      P.hsPaid += paidOf(r.m.headstart); P.blPaid += paidOf(r.m.baseline);
    }
    if (P.n) { P.hsUsd += orphan.headstart.usd; P.blUsd += orphan.baseline.usd; }   // corrected-away questions cost too
    return { T, P };
  }

  // "5.0× cheaper" / "1.3× the cost" / "same cost" (within 15 %), from session $ (paired questions)
  function costPhrase(hs, bl) {
    if (hs <= 0 && bl <= 0) return "both $0";
    if (hs <= 0) return `$0 vs ${fmtUsd(bl)}`;
    if (bl <= 0) return `${fmtUsd(hs)} vs $0`;
    const r = bl / hs;
    if (r >= 1.15) return `${x(r)} cheaper`;
    if (r <= 1 / 1.15) return `${x(1 / r)} the cost`;
    return "same cost";
  }
  function callsPhrase(hs, bl) {
    if (hs === 0 && bl === 0) return "no paid calls";
    if (hs === 0) return `0 vs ${bl} paid calls`;
    if (bl === 0) return `${hs} vs 0 paid calls`;
    const r = bl / hs;
    if (r >= 1.15) return `${x(r)} fewer paid calls`;
    if (r <= 1 / 1.15) return `${x(1 / r)} more paid calls`;
    return "same paid calls";
  }

  // ------------------------------------------------------------------ render
  function ensureRoot() {
    if (root) return root;
    root = document.createElement("section");
    root.id = "meter";
    root.className = "meter";
    root.setAttribute("aria-label", "Cost and speed meter");
    document.body.appendChild(root);
    document.body.classList.add("has-meter");
    root.addEventListener("click", (ev) => {
      const b = ev.target.closest("button");
      if (!b) return;
      if (b.classList.contains("mt-more")) { open = !open; render(); }
      else if (b.classList.contains("mt-reset")) { rows.length = 0; for (const m of MODES) orphan[m] = { in: 0, out: 0, usd: 0 }; lastRow = null; render(); }
    });
    return root;
  }

  function webNote() {
    const be = prices && prices.web_backend;
    return be && be !== "tavily" ? `web lane = ${be}: paid-lane calls are free here and priced as Tavily basic searches` : "";
  }

  function qCells(m) {
    if (!m) return `<td colspan="5" class="mt-na">not run</td>`;
    if (m.failed) return `<td colspan="5" class="mt-na">failed</td>`;
    if (!answered(m)) return `<td colspan="5" class="mt-na">…</td>`;
    const card = m.answer_by === "card";
    const tin = tokIn(m), w = m.waste_in || 0;
    const tok = card && !tin ? `<span class="mt-card" title="a live-data card answered: no model call">card · 0 tok</span>`
      : `${fmtTok(m.ans_in || 0)}<span class="mt-dim">/</span>${fmtTok(m.ans_out || 0)}` +
        (w ? ` <span class="mt-waste" title="input tokens of answer requests that were thrown away (drafts, superseded answers, hedge duplicates)">+${fmtTok(w)} waste</span>` : "") +
        (m.tokens_est ? `<span class="mt-dim" title="part of this count is estimated: a request stopped before the API reported its usage"> ~</span>` : "");
    return `<td>${paidOf(m)}</td><td>${m.free_calls ?? 0}</td><td>${m.wasted_fires != null || m.spec_fires != null ? wastedOf(m) : "—"}</td>` +
      `<td class="mt-tok">${tok}</td><td class="mt-usd">${fmtUsd(m.usd)}</td>`;
  }

  function render() {
    const el = ensureRoot();
    const { T, P } = totals();
    const modes = MODES.filter((m) => T[m]);
    if (!rows.length || !modes.length) {
      el.innerHTML = `<div class="mt-bar"><div class="mt-hero"><b class="mt-idle">Cost meter</b><small>time, paid calls, model tokens and est. $ per question</small></div></div>`;
      el.classList.remove("open");
      return;
    }
    // hero: session averages over paired questions; single mode: what was measured
    let hero, heroSub;
    if (P.n) {
      const avg = P.saved / P.n;
      const tp = Math.abs(avg) < 50 ? "same speed" : `${fmtS(Math.abs(avg))} ${avg >= 0 ? "faster" : "slower"}`;
      hero = `${tp} · ${costPhrase(P.hsUsd, P.blUsd)}`;
      heroSub = `Onset vs plain search · avg of ${P.n} question${P.n > 1 ? "s" : ""} · $ est.`;
    } else {
      const m = modes[0], t = T[m];
      hero = `${fmtUsd(t.usd / t.n)} per question`;
      heroSub = `${NAME[m]} only · ${(t.paid / t.n).toFixed(1)} paid calls/q · $ est. · run Race to compare`;
    }
    const r = lastRow;
    const qtxt = r ? (r.m.headstart || r.m.baseline)?.question || "" : "";
    const saved = r && answered(r.m.headstart) && answered(r.m.baseline) ? ready(r, "baseline") - ready(r, "headstart") : null;
    const thisQ = r ? `<table class="mt-q"><thead><tr><th title="${esc(qtxt)}">this question${saved != null ? ` · <span class="${saved >= 0 ? "mt-good" : "mt-bad"}">${fmtS(Math.abs(saved))} ${saved >= 0 ? "saved" : "lost"}</span>` : ""}</th>` +
      `<th title="calls that reach the paid search backend">paid</th><th title="card APIs, Wikidata, local index">free</th><th title="speculative calls whose results fed nothing">wasted</th>` +
      `<th title="answer-model tokens in / out of the answer, + input tokens of requests thrown away (drafts, superseded answers, hedge duplicates)">tokens</th><th>$ est.</th></tr></thead><tbody>` +
      modes.map((m) => `<tr class="mt-${SHORT[m].toLowerCase()}"><td><i class="dot ${SHORT[m].toLowerCase()}"></i>${NAME[m]}</td>${qCells(r.m[m])}</tr>`).join("") +
      `</tbody></table>` : "";
    const nQ = Math.max(...modes.map((m) => T[m].n));
    let s1, s2;
    if (P.n) {
      s1 = `<b>${nQ}</b> question${nQ > 1 ? "s" : ""} · saved <b>${fmtS(P.saved)}</b>`;
      s2 = `<b>${fmtUsd(P.hsUsd)}</b> vs <b>${fmtUsd(P.blUsd)}</b> · paid <b>${P.hsPaid}</b> vs <b>${P.blPaid}</b> · ${callsPhrase(P.hsPaid, P.blPaid)}`;
    } else {
      const t = T[modes[0]];
      s1 = `<b>${nQ}</b> question${nQ > 1 ? "s" : ""} · <b>${fmtUsd(t.usd)}</b> est.`;
      s2 = `paid <b>${t.paid}</b> · free <b>${t.free}</b> · wasted <b>${t.wasted}</b>`;
    }
    let html = `<div class="mt-bar">
      <div class="mt-hero"><b>${esc(hero)}</b><small>${esc(heroSub)}</small></div>
      ${thisQ}
      <div class="mt-sess"><label>session</label><div>${s1}</div><div>${s2}</div></div>
      <button class="mt-more" aria-expanded="${open}">${open ? "less" : "details"}</button>
    </div>`;
    if (open) html += detail(T, P, modes);
    el.innerHTML = html;
    el.classList.toggle("open", open);
  }

  function detail(T, P, modes) {
    const head = `<tr><th>#</th><th>question</th>${modes.map((m) => `<th>${SHORT[m]} ready</th>`).join("")}${modes.length > 1 ? "<th>saved</th>" : ""}` +
      modes.map((m) => `<th title="paid / free / wasted calls">${SHORT[m]} calls p/f/w</th><th title="answer-model tokens in / out of the answer, + thrown-away requests (waste)">${SHORT[m]} tokens</th><th>${SHORT[m]} $ est.</th>`).join("") + `</tr>`;
    const body = rows.slice(-12).map((r) => {
      const q = (r.m.headstart || r.m.baseline)?.question || "";
      const both = answered(r.m.headstart) && answered(r.m.baseline);
      const sv = both ? ready(r, "baseline") - ready(r, "headstart") : null;
      return `<tr><td>${r.n}</td><td class="mt-qt" title="${esc(q)}">${esc(q)}</td>` +
        modes.map((m) => `<td>${answered(r.m[m]) ? fmtS(ready(r, m)) : "—"}</td>`).join("") +
        (modes.length > 1 ? `<td class="${sv == null ? "" : sv >= 0 ? "mt-good" : "mt-bad"}">${sv == null ? "—" : fmtS(sv)}</td>` : "") +
        modes.map((m) => {
          const x = r.m[m];
          if (!answered(x)) return `<td colspan="3" class="mt-na">${x ? (x.failed ? "failed" : "—") : "not run"}</td>`;
          const card = x.answer_by === "card" && !tokIn(x);
          return `<td>${paidOf(x)} / ${x.free_calls ?? 0} / ${wastedOf(x)}</td>` +
            `<td>${card ? "card · 0" : `${fmtTok(x.ans_in || 0)} / ${fmtTok(x.ans_out || 0)}${x.waste_in ? ` <span class="mt-waste">+${fmtTok(x.waste_in)} waste</span>` : ""}${x.tokens_est ? " ~" : ""}`}</td>` +
            `<td>${fmtUsd(x.usd)}</td>`;
        }).join("") + `</tr>`;
    }).join("");
    const tot = modes.map((m) => {
      const t = T[m];
      return `<span><i class="dot ${SHORT[m].toLowerCase()}"></i><b>${NAME[m]}</b> ${t.n} q · paid ${t.paid} · free ${t.free} · wasted ${t.wasted} · tokens ${fmtTok(t.tin)} / ${fmtTok(t.tout)}` +
        (t.win ? ` (${fmtTok(t.win)} in thrown away)` : "") + ` · <b>${fmtUsd(t.usd)}</b> est.${t.cards ? ` · ${t.cards} card answer${t.cards > 1 ? "s" : ""}` : ""}</span>`;
    }).join("");
    const p = prices;
    const priceTxt = p ? `Prices (est.): Tavily ${fmtUsd(p.credit_usd)}/credit × ${p.credits_per_call} credit per basic search; ` +
      (p.in_usd_mtok != null ? `${esc(p.model)} $${p.in_usd_mtok} in / $${p.out_usd_mtok} out per M tokens${p.verified ? "" : " (configured, not verified)"}` : `${esc(p.model)}: no verified price, tokens not priced`) : "";
    const orph = MODES.filter((m) => orphan[m].in).map((m) => `${SHORT[m]} ${fmtTok(orphan[m].in)} tokens on questions replaced by a correction`).join(" · ");
    return `<div class="mt-detail">
      <div class="mt-tot">${tot}</div>
      <div class="mt-tbl"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>
      <p class="mt-note">Ready = answer's first word after the last spoken word. Paid = calls that reach the paid search backend; free = card APIs, Wikidata, local index; wasted = speculative calls whose results fed nothing.
      Tokens = answer model, input / output, including thrown-away drafts, superseded answers and hedge duplicates ("waste"); ~ = partly estimated (request stopped before the API reported usage); card answers make no model call.
      ${esc(priceTxt)}${webNote() ? ". " + esc(webNote()) : ""}.${orph ? " " + esc(orph) + "." : ""}
      <button class="link-btn mt-reset">reset session</button></p>
    </div>`;
  }

  // ------------------------------------------------------------------ eval A/B summary (Results section)
  function evalSummary(raw) {
    const rs = (raw && (raw.rows || raw.results)) || [];
    if (!Array.isArray(rs) || !rs.length) return "";
    const by = {};
    for (const r of rs) if (r && MODES.includes(r.mode)) (by[r.mode] = by[r.mode] || []).push(r);
    const ms = MODES.filter((m) => by[m]);
    if (!ms.length) return "";
    const p = prices || { credit_usd: 0.008, credits_per_call: 1, in_usd_mtok: null, out_usd_mtok: null };
    const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
    const S = {};
    for (const m of ms) {
      const g = by[m];
      const hasTok = g.some((r) => r.ans_in != null);
      const paid = mean(g.map((r) => r.paid_calls ?? r.web_paid ?? r.web_calls ?? 0));
      const cr = mean(g.map((r) => (r.credits ? r.credits : (r.paid_calls ?? r.web_paid ?? r.web_calls ?? 0) * p.credits_per_call)));
      const tin = hasTok ? mean(g.map((r) => (r.ans_in || 0) + (r.waste_in || 0))) : null;
      const tout = hasTok ? mean(g.map((r) => (r.ans_out || 0) + (r.waste_out || 0))) : null;
      const llm = hasTok && p.in_usd_mtok != null ? (tin * p.in_usd_mtok + tout * p.out_usd_mtok) / 1e6 : null;
      S[m] = { n: g.length, paid, free: mean(g.map((r) => r.free_calls || 0)),
        wasted: mean(g.map((r) => Math.max(0, (r.spec_fires || 0) - (r.spec_used || 0)))),
        tin, tout, waste: hasTok ? mean(g.map((r) => r.waste_in || 0)) : null,
        usd: cr * p.credit_usd + (llm || 0), hasTok,
        ready: (() => { const a = g.map((r) => r.ready_ms).filter((v) => v != null).sort((a, b) => a - b); return a.length ? a[Math.floor((a.length - 1) / 2)] : null; })() };
    }
    const cell = (v, f) => (v == null ? "—" : f(v));
    const n2 = (v) => v.toFixed(2);
    let h = `<div class="mt-eval"><h3>Cost &amp; speed, same terms as the live meter <span class="sub">${esc(raw.created || "")} · per question · $ est.</span></h3><table><thead><tr><th></th><th>p50 ready</th><th>paid</th><th>free</th><th>wasted</th><th>tokens in / out</th><th>$ est. / q</th><th>$ est. / 1,000 q</th></tr></thead><tbody>`;
    for (const m of ms) {
      const s = S[m];
      h += `<tr><td><i class="dot ${SHORT[m].toLowerCase()}"></i>${NAME[m]} <span class="muted">n=${s.n}</span></td><td>${fmtS(s.ready)}</td><td>${n2(s.paid)}</td><td>${n2(s.free)}</td><td>${m === "baseline" ? "—" : n2(s.wasted)}</td>` +
        `<td>${s.hasTok ? `${fmtTok(s.tin)} / ${fmtTok(s.tout)}${s.waste ? ` <span class="mt-waste">(${fmtTok(s.waste)} waste)</span>` : ""}` : `<span class="muted" title="this A/B file was written before the meter recorded tokens">not recorded</span>`}</td>` +
        `<td>${fmtUsd(s.usd)}${s.hasTok ? "" : "*"}</td><td>${cell(s.usd * 1000, (v) => "$" + v.toFixed(2))}</td></tr>`;
    }
    h += `</tbody></table>`;
    if (S.headstart && S.baseline) {
      const a = S.headstart, b = S.baseline;
      const bits = [];
      if (a.ready != null && b.ready != null) bits.push(`p50 ${fmtS(b.ready - a.ready)} ${b.ready >= a.ready ? "faster" : "slower"}`);
      bits.push(costPhrase(a.usd, b.usd));
      bits.push(callsPhrase(+a.paid.toFixed(2), +b.paid.toFixed(2)).replace(/(\d+\.\d{3,})/g, (s) => (+s).toFixed(2)));
      h += `<p class="res-note">Onset vs plain search: <b>${esc(bits.join(" · "))}</b></p>`;
    }
    if (ms.some((m) => !S[m].hasTok)) h += `<p class="res-note muted">* web calls only: this file has no answer-model token counts (recorded before the meter); the model's share at list price is well under a cent per question.</p>`;
    const be = rs.find((r) => r.web_backend)?.web_backend;
    if (be && be !== "tavily") h += `<p class="res-note muted">Web lane in this run: ${esc(be)} (free); paid calls are priced as Tavily basic searches.</p>`;
    return h + `</div>`;
  }

  // ------------------------------------------------------------------ API
  async function loadPrices() {
    try { prices = (await (await fetch("/api/meter", { cache: "no-store" })).json()) || null; } catch { prices = null; }
  }
  window.HSMeter = {
    start(run) { ensureRoot(); render(); },
    metric(q, mode, e, run) {
      if (!MODES.includes(mode)) return;
      const r = rowFor(q, run);
      const prev = r.m[mode];
      r.m[mode] = { ...e };
      if (prev && prev._usage && prev.q === e.q) for (const k of TOK_KEYS) if (prev._usage[k] !== undefined) r.m[mode][k] = prev._usage[k];
      if (answered(e)) lastRow = r;
      render();
    },
    event(e, run) {
      // final token tally of a closed question ("usage", after its metric): replaces the metric's token / $ fields
      if (e.t !== "usage" || !run || !MODES.includes(e.mode)) return;
      const r = e.answered ? rows.find((o) => o.run === run && o.m[e.mode] && o.m[e.mode].q === e.q) : null;
      if (r) {
        const m = r.m[e.mode];
        for (const k of TOK_KEYS) if (e[k] !== undefined) m[k] = e[k];
        m._usage = e;
        render();
      } else {
        // a question replaced by a correction (no metric of its own): its spend still counts
        orphan[e.mode].in += (e.ans_in || 0) + (e.waste_in || 0);
        orphan[e.mode].out += (e.ans_out || 0) + (e.waste_out || 0);
        orphan[e.mode].usd += e.usd || 0;
        if (rows.length) render();
      }
    },
    state() { const { T, P } = totals(); return { T, P, rows: rows.map((r) => ({ n: r.n, hs: r.m.headstart || null, bl: r.m.baseline || null })), orphan, prices }; },
    evalSummary,
  };
  loadPrices();
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => render());
  else render();
})();
