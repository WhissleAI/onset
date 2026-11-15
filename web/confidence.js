// One trust line under each answer: the calibrated confidence (server/calibrate.py, "confidence" event) and the
// post-answer source check (server/verify.py "verify", server/recheck.py "recheck") merged into ONE chip in plain
// words, e.g. "High confidence · 2 sources agree" / "Unverified · sources don't cover this". The numbers and the
// per-claim detail are in the tooltip. app.js calls HSConf.event(e) for every event and HSConf.reset() on a new question;
// web/plan.js hands its verify / recheck events here (HSConf.verify / HSConf.recheck).
(function () {
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const WORD = { high: "High confidence", mid: "Medium confidence", low: "Low confidence" };
  const st = {};                 // mode -> {conf, verify, recheck}

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

  // the source check in plain words (no glyphs, no counts of "claims")
  function verifyWords(v) {
    if (!v) return "";
    const n = String(v.label || "").match(/(\d+) sources agree/);
    if (v.status === "agree") return `${n ? n[1] : 2} sources agree`;
    if (v.status === "supported") return "matches a source";
    if (v.status === "disagree") return String(v.label || "sources disagree").replace(/^\W+\s*/, "");
    if (v.status === "partial") return "partly found in sources";
    return "";
  }

  // {band, word, detail} of the merged line; null when there is nothing to say yet
  function merge(s) {
    const c = s.conf, v = s.verify, r = s.recheck;
    const vOk = v && (v.status === "agree" || v.status === "supported");
    let band, word, detail;
    if (c && c.conf != null) {
      band = WORD[c.band] ? c.band : "mid";
      word = WORD[band];
      detail = c.reason || "";
      if (/sources don't say/.test(detail)) { band = "low"; word = "Unverified"; detail = "sources don't cover this"; }
      else if (band === "high" && vOk) detail = verifyWords(v);          // the check's count beats a generic "matches"
      else if (band !== "high" && v && v.status === "disagree") detail = verifyWords(v);
      else if (!detail && band === "low") detail = "check this";
    } else if (v) {
      // no calibration model installed: the source check alone
      if (vOk) { band = "high"; word = "Checked"; detail = verifyWords(v); }
      else if (v.status === "disagree") { band = "low"; word = "Check this"; detail = verifyWords(v); }
      else if (v.status === "partial") { band = "mid"; word = "Partly checked"; detail = ""; }
      else { band = "mid"; word = "Unverified"; detail = ""; }
    } else return null;
    if (r && r.status === "rechecked") detail = (detail ? detail + ", " : "") + "re-checked";
    if (r && r.status === "corrected") { band = "low"; word = "Corrected"; detail = "see the note above"; }
    return { band, word, detail };
  }

  function tip(s) {
    const out = [];
    const c = s.conf, v = s.verify, r = s.recheck;
    if (c && c.conf != null) {
      out.push(`Answers like this were right about ${Math.round(c.conf * 100)}% of the time on our test questions ` +
        `(calibrated on labelled answers; Wikipedia search).` + (c.reason ? ` Main signal: ${c.reason}.` : ""));
    }
    if (v) {
      out.push(`Source check: ${String(v.label || v.status).replace(/^\W+\s*/, "")} (numbers, years and names matched ` +
        `against the cited sources${v.ms != null ? ", " + v.ms + " ms" : ""}; it checks that the sources say it, not that it is true).`);
      for (const cl of v.claims || []) {
        out.push(`  ${cl.text}: ${cl.support && cl.support.length ? "in [" + cl.support.join("], [") + "]" : "not found"}` +
          (cl.conflict ? ` (source [${cl.conflict.source}] says ${cl.conflict.value})` : ""));
      }
    }
    if (r && r.label) {
      out.push(`Re-checked "${r.claim || ""}" after the answer: ${(r.lanes || []).join(" → ") || "no lane"}` +
        (r.paid ? " (1 paid call)" : " (free)") + (r.ms != null ? `, ${r.ms} ms` : "") + ` · ${r.label}`);
    }
    return out.join("\n");
  }

  function paint(mode) {
    const s = st[mode];
    const b = box(mode, !!s);
    if (!b) return;
    const m = s && merge(s);
    if (!m) { b.innerHTML = ""; return; }
    b.innerHTML = `<span class="cb cb-${m.band}" title="${esc(tip(s))}"><i class="cdot"></i><b>${esc(m.word)}</b>` +
      `${m.detail ? `<span class="cd"> · ${esc(m.detail)}</span>` : ""}</span>`;
  }

  function put(mode, k, v) {
    if (!mode || mode === "sys" || mode === "ear") return;
    (st[mode] = st[mode] || {})[k] = v;
    paint(mode);
  }

  function clear(mode) {
    for (const m of mode ? [mode] : ["headstart", "baseline"]) {
      delete st[m];
      const b = box(m, false);
      if (b) b.innerHTML = "";
    }
  }

  window.HSConf = {
    event(e) {
      if (e.mode === "sys" || e.mode === "ear") return;
      if (e.t === "confidence") put(e.mode, "conf", e);
      else if (e.t === "extended") clear(e.mode);
    },
    verify(e) { put(e.mode, "verify", e); },
    recheck(e) { if (e.label && e.status !== "skip") put(e.mode, "recheck", e); },
    reset() { clear(); },
    _merge: merge,               // tests (tests/test_trust_line.py)
  };
})();
