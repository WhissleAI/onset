// Plan while you speak + verify (server/planner.py, server/verify.py).
//  - "plan" events: the sub-questions of a multi-part question, shown as chips in the query-frame card while the user
//    speaks (P1 / P2 / P3, each with its lane and status: waiting · searching · done · stale).
//  - "verify" events: the answer's claims checked against its sources, shown as a badge under each pane's answer.
// app.js calls HSPlan.event(e) for every event, HSPlan.reset() on a new question, HSPlan.partNo() for timeline labels.
(function () {
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const plans = {};            // mode -> last plan event
  const ids = {};              // mode -> part text -> "P1"
  const STATE = { waiting: "waiting", fired: "searching", done: "done", stale: "stale", skipped: "skipped",
    error: "error", cancelled: "done" };
  const TIER = { card: "card", facts: "facts", web: "web", wait: "facts" };

  function row() {
    // under the live transcript ("What Arc hears"): visible in the minimal (demo) view and in details
    let r = document.getElementById("fPlanRow");
    if (r) return r;
    const tr = document.getElementById("transcript");
    if (!tr) return null;
    r = document.createElement("div");
    r.className = "plan-strip";
    r.id = "fPlanRow";
    r.hidden = true;
    r.innerHTML = '<span class="plab" title="a multi-part question: each part is searched on its own, while you speak">plan</span><div class="planchips" id="fPlan"></div>';
    tr.after(r);
    return r;
  }

  function badgeBox(mode) {
    const pane = document.getElementById("pane-" + mode);
    if (!pane) return null;
    let b = pane.querySelector(".verify");
    if (!b) {
      b = document.createElement("div");
      b.className = "verify";
      pane.querySelector('[data-k="answer"]').after(b);
    }
    return b;
  }

  function paint() {
    const r = row();
    if (!r) return;
    const mode = plans.headstart && plans.headstart.parts.length ? "headstart" : (plans.baseline && plans.baseline.parts.length ? "baseline" : null);
    if (!mode) { r.hidden = true; return; }
    const p = plans[mode];
    r.hidden = false;
    const box = r.querySelector("#fPlan");
    const html = p.parts.map((x) => {
      const st = x.state && x.state.startsWith("needs") ? "waiting" : (x.state || "waiting");
      const lane = TIER[x.tier] || "";
      const title = `${x.id.toUpperCase()} · ${x.kind}${x.closed ? " · closed (you moved past it)" : ""} · ${STATE[st] || st}` +
        (x.n ? ` · ${x.n} docs` : "") + (x.state && x.state.startsWith("needs") ? " · waits for P1's answer" : "");
      return `<span class="pchip st-${esc(st)}" title="${esc(title)}"><b>${esc(x.id.toUpperCase())}</b>` +
        `<span class="pq">${esc(x.text)}</span>${lane ? `<i class="pl pl-${lane}">${lane}</i>` : ""}</span>`;
    }).join("") + `<span class="pwhy">${esc(p.kind || "")}</span>`;
    if (box._h !== html) { box.innerHTML = html; box._h = html; }
  }

  function onPlan(e) {
    plans[e.mode] = e;
    const m = (ids[e.mode] = ids[e.mode] || {});
    for (const x of e.parts || []) m[x.text] = x.id.toUpperCase();
    paint();
  }

  function onVerify(e) {
    const b = badgeBox(e.mode);
    if (!b) return;
    const cls = { agree: "ok", supported: "ok", disagree: "warn", partial: "part", unverified: "none" }[e.status] || "none";
    const claims = (e.claims || []).map((c) => `${c.text}: ${c.support && c.support.length ? "in [" + c.support.join("], [") + "]" : "not found"}` +
      (c.conflict ? ` (source [${c.conflict.source}] says ${c.conflict.value})` : "")).join("\n");
    b.innerHTML = `<span class="vb ${cls}" title="${esc("Checked after the answer, against its sources (numbers, years, names; string matching, " + e.ms + " ms). It checks that the sources say it, not that it is true.\n" + claims)}">${esc(e.label)}</span>`;
  }

  function clear(mode) {
    const modes = mode ? [mode] : ["headstart", "baseline"];
    for (const m of modes) {
      delete plans[m];
      delete ids[m];
      const pane = document.getElementById("pane-" + m);
      const b = pane && pane.querySelector(".verify");
      if (b) b.innerHTML = "";
    }
    paint();
  }

  window.HSPlan = {
    event(e) {
      if (e.mode === "sys" || e.mode === "ear") return;
      if (e.t === "plan") onPlan(e);
      else if (e.t === "verify") onVerify(e);
      else if (e.t === "extended") { const b = badgeBox(e.mode); if (b) b.innerHTML = ""; }
    },
    reset() { clear(); },
    partNo(mode, text) { return (ids[mode] || {})[text] || ""; },
  };
})();
