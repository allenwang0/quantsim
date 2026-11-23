/* report-kit v1.0.6 — behaviour helpers. Canonical: report-design-system/kit/report.js
   No dependencies. Load with a deferred script tag pointing at kit/report.js.
   Theme: put the one-line theme script (see README) in <head> BEFORE css so there's no flash. */
(function () {
  "use strict";
  var RK = (window.RK = window.RK || {});

  /* Sticky bar: hairline after scroll, title after hero H1 leaves viewport. */
  RK.initBar = function () {
    var bar = document.querySelector(".bar");
    if (!bar) return;
    var onScroll = function () { bar.classList.toggle("scrolled", window.scrollY > 8); };
    addEventListener("scroll", onScroll, { passive: true }); onScroll();
    var h1 = document.querySelector("main h1");
    if (h1 && "IntersectionObserver" in window)
      new IntersectionObserver(function (e) { bar.classList.toggle("past-hero", !e[0].isIntersecting); }).observe(h1);
  };

  /* Segmented control: <div class="seg" role="radiogroup"><button role="radio" aria-checked data-value>…
     cb(value, button). Arrow keys move; URL param optional (opts.param) via replaceState. */
  RK.segmented = function (el, cb, opts) {
    if (typeof el === "string") el = document.querySelector(el);
    if (!el) return;
    opts = opts || {};
    var bs = Array.prototype.slice.call(el.querySelectorAll('[role="radio"]'));
    function pick(b, fromUrl) {
      bs.forEach(function (x) { var on = x === b; x.setAttribute("aria-checked", on); x.tabIndex = on ? 0 : -1; });
      if (opts.param && !fromUrl) try { var u = new URL(location.href); u.searchParams.set(opts.param, b.dataset.value); history.replaceState(null, "", u); } catch (e) {}
      cb(b.dataset.value, b);
    }
    bs.forEach(function (b, i) {
      b.addEventListener("click", function () { pick(b); });
      b.addEventListener("keydown", function (e) {
        var j = e.key === "ArrowRight" || e.key === "ArrowDown" ? i + 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? i - 1 : null;
        if (j === null) return; e.preventDefault(); j = (j + bs.length) % bs.length; bs[j].focus(); pick(bs[j]);
      });
    });
    var init = null;
    if (opts.param) try { var q = new URLSearchParams(location.search).get(opts.param); init = bs.filter(function (b) { return b.dataset.value === q; })[0]; } catch (e) {}
    pick(init || bs.filter(function (b) { return b.getAttribute("aria-checked") === "true"; })[0] || bs[0], true);
  };

  /* Chart/Table toggle: a .seg with data-value="chart"|"table" inside a .figure that holds .plot and .rk-table */
  RK.viewToggles = function () {
    document.querySelectorAll(".figure [data-view-toggle]").forEach(function (seg) {
      var fig = seg.closest(".figure");
      RK.segmented(seg, function (v) {
        fig.querySelectorAll(".plot, .figure > img, .fig-img").forEach(function (p) { p.hidden = v !== "chart"; });
        fig.querySelectorAll(".rk-table").forEach(function (t) { t.hidden = v !== "table"; });
      });
    });
  };

  /* Copy buttons on <div class="codewrap"><pre><code>… */
  RK.copyButtons = function () {
    document.querySelectorAll(".codewrap").forEach(function (w) {
      if (w.querySelector(".copy") || !navigator.clipboard) return;
      var b = document.createElement("button"); b.className = "copy"; b.type = "button"; b.textContent = "Copy";
      b.setAttribute("aria-live", "polite");
      b.addEventListener("click", function () {
        var t = w.querySelector("code").innerText.split("\n").map(function (l) { return l.replace(/\s+#.*$/, ""); }).join("\n");
        navigator.clipboard.writeText(t).then(function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1500); });
      });
      w.appendChild(b);
    });
  };

  /* Read <script type="application/json" id="..."> */
  RK.data = function (id) { var el = document.getElementById(id); return el ? JSON.parse(el.textContent) : null; };

  /* Number formatting per spec §21 */
  RK.fmt = {
    sig: function (v, n) { if (v === 0) return "0"; n = n || 3; var d = Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(v)))); return v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: Math.min(d, 2) > 0 && Math.abs(v) < 1000 ? 0 : 0 }); },
    compact: function (v, unit) { var a = Math.abs(v), s = a >= 1e9 ? [v / 1e9, " B"] : a >= 1e6 ? [v / 1e6, " M"] : a >= 1e4 ? [v / 1e3, " K"] : [v, ""]; return (+s[0].toPrecision(3)).toLocaleString("en-US") + s[1] + (unit ? " " + unit : ""); },
    ratio: function (r) { return (r < 10 ? r.toFixed(2) : r < 100 ? r.toFixed(1) : Math.round(r)) + "×"; },
    pct: function (p) { var a = Math.abs(p); return (a >= 10 ? Math.round(p) : p.toFixed(1)) + " %"; }
  };

  function init() { RK.initBar(); RK.viewToggles(); RK.copyButtons(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
