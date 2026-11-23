/* quantsim report: draws the charts from the embedded #results JSON and switches strategy.
   Every value comes from results.json; nothing here is hand-typed. */
(function () {
  "use strict";
  function num(v, d) { var t = (+v).toFixed(d == null ? 2 : d); return v < 0 ? t.replace("-", "−") : t; }
  function yearFrac(s) { var d = new Date(s + "T00:00:00Z"), y = d.getUTCFullYear(), a = Date.UTC(y, 0, 1), b = Date.UTC(y + 1, 0, 1); return y + (d - a) / (b - a); }
  function draw(results) {
    var S = results.strategies || {}, names = Object.keys(S), blue = "var(--series-1)", orange = "var(--series-2)";
    if (names.length) {
      RKCharts.dotRange("#c-hero", {
        series: [{ key: "s", name: "Out-of-sample Sharpe", color: blue }],
        rows: names.map(function (n) { var u = S[n].sharpe_uncertainty; return { label: S[n].label, series: { s: { lo: u.ci_low, mid: u.point, hi: u.ci_high, tip: num(u.point) + " [" + num(u.ci_low) + ", " + num(u.ci_high) + "]" } } }; }),
        refs: [{ x: 0, label: "No edge" }], xLabel: "Annualized Sharpe ratio", format: function (v) { return num(v, 1); },
        title: names.map(function (n) { var u = S[n].sharpe_uncertainty; return S[n].label + ": Sharpe " + u.point + ", 95% interval " + u.ci_low + " to " + u.ci_high; }).join(". ")
      });
    }
    names.forEach(function (n) {
      var wf = S[n].walk_forward;
      RKCharts.dotRange("#c-win-" + n, {
        series: [{ key: "s", name: "Sharpe", color: blue }],
        rows: wf.windows.map(function (w) { return { label: w.test_start.slice(0, 4) + "–" + w.test_end.slice(2, 4), series: { s: { lo: w.ci_low, mid: w.oos_sharpe, hi: w.ci_high, tip: num(w.oos_sharpe) + " [" + num(w.ci_low) + ", " + num(w.ci_high) + "]" } } }; }),
        refs: [{ x: 0, label: "No edge" }], xLabel: "Annualized Sharpe ratio", format: function (v) { return v === Math.round(v) ? num(v, 0) : num(v, 1); },
        title: S[n].label + ": out-of-sample Sharpe per test window with 95% intervals"
      });
      RKCharts.line("#c-eq-" + n, {
        x: wf.dates.map(yearFrac),
        series: [{ key: "s", name: S[n].label, color: blue, y: wf.oos_equity }, { key: "b", name: "Buy-and-hold", color: orange, y: wf.buy_and_hold, dash: true }],
        yLabel: "Equity (USD)", yUnit: "USD", zero: false, endLabels: false,
        xFormat: function (v) { return String(Math.round(v)); },
        xTicks: (function () { var a = [], y0 = Math.ceil(yearFrac(wf.dates[0])), y1 = Math.floor(yearFrac(wf.dates[wf.dates.length - 1])); for (var y = y0; y <= y1; y++) a.push(y); return a; })(),
        format: function (v) { return Math.abs(v) >= 1000 ? "$" + Math.round(v / 1000) + "k" : "$" + Math.round(v); },
        title: S[n].label + " out-of-sample equity " + wf.oos_return_pct + "% vs buy-and-hold " + wf.buy_and_hold_return_pct + "%"
      });
    });
    var b = results.benchmark;
    if (b && document.getElementById("c-speed")) {
      RKCharts.line("#c-speed", {
        x: b.rows.map(function (r) { return r.assets; }),
        xTicks: b.rows.map(function (r) { return r.assets; }),
        series: [{ key: "v", name: "Vectorized", color: blue, y: b.rows.map(function (r) { return r.vectorized_median_s; }) }, { key: "e", name: "Event-driven", color: orange, y: b.rows.map(function (r) { return r.event_median_s; }) }],
        xLabel: "Assets", yLabel: "Median seconds (log scale)", yUnit: "s", log: true, height: 280,
        format: function (v) { return v >= 1 ? String(+v.toPrecision(3)) : String(+v.toPrecision(2)); },
        title: "Median seconds by asset count: " + b.rows_shown.map(function (r) { return r.assets + " assets vectorized " + r.vectorized_s_shown + " s, event " + r.event_s_shown + " s"; }).join("; ")
      });
    }
  }
  function init() {
    var results = RK.data("results");
    if (!results) return;
    draw(results);
    document.querySelectorAll(".figure .rk-table").forEach(function (t) { t.hidden = true; });
    var pick = document.getElementById("strategy");
    if (pick) RK.segmented(pick, function (v) {
      document.querySelectorAll("[data-strategy]").forEach(function (el) { el.hidden = el.getAttribute("data-strategy") !== v; });
    }, { param: "strategy" });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
