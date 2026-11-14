// Answer confidence badge (server/calibrate.py, CALIB_BADGE): one calibrated confidence per answer and a plain reason,
// e.g. "High confidence · 2 sources agree" / "Low · check this". Sits next to the verify badge under each answer.
// app.js calls HSConf.event(e) for every event and HSConf.reset() on a new question.
(function () {
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const WORD = { high: "High confidence", mid: "Medium", low: "Low" };

  function box(mode, make) {
    const pane = document.getElementById("pane-" + mode);
    if (!pane) return null;
    let b = pane.querySelector(".confb");
    if (!b && make) {
      const ans = pane.querySelector('[data-k="answer"]');
      if (!ans) return null;
      b = document.createElement("div");
      b.className = "confb";
      ans.after(b);
    }
    return b;
  }

  function show(e) {
    const b = box(e.mode, true);
    if (!b || e.conf == null) return;
    const band = WORD[e.band] ? e.band : "mid";
    const reason = e.band === "low" && !e.reason ? "check this" : (e.reason || "");
    const pct = Math.round(e.conf * 100);
    const tip = `Answers like this were right about ${pct}% of the time on our test questions ` +
      `(calibrated on labelled answers; Wikipedia search). ${reason ? "Main signal: " + reason + "." : ""}`;
    b.innerHTML = `<span class="cb cb-${band}" title="${esc(tip)}"><i class="cdot"></i>${esc(WORD[band])}` +
      `${reason ? ` · ${esc(reason)}` : ""}</span>`;
  }

  function clear(mode) {
    for (const m of mode ? [mode] : ["headstart", "baseline"]) {
      const b = box(m, false);
      if (b) b.innerHTML = "";
    }
  }

  window.HSConf = {
    event(e) {
      if (e.mode === "sys" || e.mode === "ear") return;
      if (e.t === "confidence") show(e);
      else if (e.t === "extended") clear(e.mode);
    },
    reset() { clear(); },
  };
})();
