/* report-kit v1.0.0 — charts. Canonical: /Users/allenwang/Desktop/report-design-system/kit/charts.js
   Responsive SVG drawn at real pixel width (text never scales down), redrawn on resize.
   Colors come from CSS tokens (var(--series-1) …), so light/dark work automatically.
   Every chart: pass `table` target to also render an HTML table twin, and `title` for the SVG <desc>.

   RKCharts.hbar(el, {rows:[{label, values:{seriesKey:number}, err?:{seriesKey:[lo,hi]}}], series:[{key,name,color}], unit, format, title, zero:true})
   RKCharts.dotRange(el, {rows:[{label, series:{key:{lo,mid,hi,mark?}}}], series, unit, ticks?:{step,label}, xmin?, xmax?, log?, title})
   RKCharts.line(el, {x:[...], series:[{key,name,color,y:[...],lo?:[...],hi?:[...]}], xLabel, yLabel, yUnit, zero?, log?, refs?:[{y,label}], title})
   RKCharts.table(el, {caption, columns:[{key,label,align}], rows:[{...}]})
*/
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var C = (window.RKCharts = {});
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function nice(max) { if (max <= 0) return 1; var e = Math.pow(10, Math.floor(Math.log10(max))), f = max / e; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * e; }
  function ticks(min, max, n) { var step = nice((max - min) / (n || 5)) || 1, out = [], v = Math.floor(min / step + 1e-9) * step; out.push(+v.toPrecision(12)); while (v < max - step * 1e-9) { v += step; out.push(+v.toPrecision(12)); } return out; }
  function logTicks(min, max) { var out = []; for (var e = Math.floor(Math.log10(min)); Math.pow(10, e) <= max * 1.0001; e++) out.push(Math.pow(10, e)); return out; }
  function deffmt(v) { var a = Math.abs(v); return a >= 1e9 ? +(v / 1e9).toPrecision(3) + "B" : a >= 1e6 ? +(v / 1e6).toPrecision(3) + "M" : a >= 1e4 ? +(v / 1e3).toPrecision(3) + "K" : (+v.toPrecision(3)).toLocaleString("en-US"); }
  function width(el) { return Math.max(260, el.clientWidth || el.parentNode.clientWidth || 600); }
  function mount(el, draw) {
    if (typeof el === "string") el = document.querySelector(el);
    if (!el) return;
    el.classList.add("plot");
    var last = 0, t;
    function go() { var w = width(el); if (w === last) return; last = w; draw(el, w); }
    go();
    if ("ResizeObserver" in window) new ResizeObserver(function () { clearTimeout(t); t = setTimeout(go, 100); }).observe(el);
    return el;
  }
  function tip(el) {
    var d = el.querySelector(".plot-tip");
    if (!d) { d = document.createElement("div"); d.className = "plot-tip"; d.setAttribute("role", "tooltip"); d.hidden = true; el.appendChild(d); }
    return {
      show: function (html, x, y) { d.innerHTML = html; d.hidden = false; var w = d.offsetWidth, W = el.clientWidth; d.style.left = Math.min(Math.max(0, x - w / 2), W - w) + "px"; d.style.top = Math.max(0, y - d.offsetHeight - 10) + "px"; },
      hide: function () { d.hidden = true; }
    };
  }
  function bindTip(el, svg) {
    var tp = tip(el);
    svg.querySelectorAll("[data-tip]").forEach(function (g) {
      g.setAttribute("tabindex", "0");
      var on = function (e) { var r = el.getBoundingClientRect(), b = g.getBoundingClientRect(); tp.show(g.getAttribute("data-tip"), (e && e.clientX ? e.clientX : b.left + b.width / 2) - r.left, b.top - r.top); };
      g.addEventListener("mousemove", on); g.addEventListener("focus", on);
      g.addEventListener("mouseleave", tp.hide); g.addEventListener("blur", tp.hide);
      g.addEventListener("click", on);
      g.addEventListener("keydown", function (e) { if (e.key === "Escape") tp.hide(); });
    });
  }
  function svgOpen(w, h, title) { return '<svg xmlns="' + NS + '" class="chart" viewBox="0 0 ' + w + " " + h + '" width="' + w + '" height="' + h + '" role="img"' + (title ? ' aria-label="' + esc(title) + '"' : "") + ">"; }
  function legend(series) {
    if (!series || series.length < 2) return "";
    return '<div class="legend" aria-hidden="true">' + series.map(function (s) { return '<span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="' + s.color + '"/></svg>' + esc(s.name) + "</span>"; }).join("") + "</div>";
  }

  /* ---------- horizontal bars ---------- */
  C.hbar = function (target, o) {
    var series = o.series || [{ key: "v", name: o.unit || "", color: "var(--series-1)" }], fmt = o.format || deffmt;
    return mount(target, function (el, W) {
      var narrow = W < 480, labelW = o.labelWidth || (narrow ? 0 : Math.min(200, W * 0.28)), barH = Math.min(o.barHeight || 20, 24), gap = 4, groupGap = narrow ? 14 : 16;
      var lab = narrow ? 18 : 0, rowH = lab + series.length * (barH + gap) - gap + groupGap, top = 4, axisH = 34, H = top + o.rows.length * rowH + axisH;
      var max = 0; o.rows.forEach(function (r) { series.forEach(function (s) { var v = r.values[s.key]; if (r.err && r.err[s.key]) v = Math.max(v, r.err[s.key][1]); if (v > max) max = v; }); });
      var tk = ticks(0, o.xmax || max * 1.02, narrow ? 3 : 5), xmax = tk[tk.length - 1], valW = narrow ? 64 : 84;
      var x0 = labelW, x1 = W - valW, x = function (v) { return x0 + (x1 - x0) * v / xmax; };
      var s = svgOpen(W, H, o.title);
      tk.forEach(function (t) { s += '<line x1="' + x(t) + '" x2="' + x(t) + '" y1="' + top + '" y2="' + (H - axisH) + '" stroke="var(--grid)"/><text x="' + x(t) + '" y="' + (H - axisH + 16) + '" text-anchor="middle">' + fmt(t) + "</text>"; });
      if (o.unit) s += '<text x="' + ((x0 + x1) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.unit) + "</text>";
      o.rows.forEach(function (r, i) {
        var y = top + i * rowH;
        if (narrow) s += '<text class="op" x="0" y="' + (y + 12) + '">' + esc(r.label) + "</text>";
        else s += '<text class="op" x="' + (x0 - 10) + '" y="' + (y + (series.length * (barH + gap) - gap) / 2 + 4) + '" text-anchor="end">' + esc(r.label) + "</text>";
        series.forEach(function (se, j) {
          var v = r.values[se.key]; if (v == null) return;
          var by = y + lab + j * (barH + gap), w = Math.max(0, x(v) - x0), rr = Math.min(4, w / 2);
          var d = "M" + x0 + "," + by + "h" + (w - rr) + "a" + rr + "," + rr + " 0 0 1 " + rr + "," + rr + "v" + (barH - 2 * rr) + "a" + rr + "," + rr + " 0 0 1 " + -rr + "," + rr + "h" + -(w - rr) + "z";
          s += '<g data-tip="' + esc("<b>" + r.label + "</b><br>" + (series.length > 1 ? se.name + ": " : "") + fmt(v) + (o.unit ? " " + o.unit : "")) + '"><path d="' + d + '" fill="' + se.color + '"/>';
          var e = r.err && r.err[se.key];
          if (e) s += '<line x1="' + x(e[0]) + '" x2="' + x(e[1]) + '" y1="' + (by + barH / 2) + '" y2="' + (by + barH / 2) + '" stroke="var(--ink-1)" stroke-width="1.5"/>';
          s += '<text class="val" x="' + (Math.max(x(v), e ? x(e[1]) : 0) + 6) + '" y="' + (by + barH / 2 + 4) + '">' + fmt(v) + "</text></g>";
        });
      });
      el.innerHTML = legend(series) + s + "</svg>";
      bindTip(el, el.querySelector("svg.chart"));
    });
  };

  /* ---------- dot-and-range (percentiles / intervals) ---------- */
  C.dotRange = function (target, o) {
    var series = o.series, fmt = o.format || deffmt;
    return mount(target, function (el, W) {
      var narrow = W < 480, L = narrow ? 0 : Math.min(150, W * 0.24), R = 16, lab = narrow ? 18 : 0, laneH = 14, rowH = lab + series.length * laneH + 18, top = 6, B = 40;
      var H = top + o.rows.length * rowH + B, lo = Infinity, hi = -Infinity;
      o.rows.forEach(function (r) { series.forEach(function (s) { var p = r.series[s.key]; if (!p) return; lo = Math.min(lo, p.lo, p.mid); hi = Math.max(hi, p.hi, p.mid); }); });
      var xmin = o.xmin != null ? o.xmin : (o.log ? lo : Math.min(0, lo)), xmax = o.xmax != null ? o.xmax : hi * 1.05, tk, x;
      if (o.log) { tk = logTicks(xmin, xmax); xmin = tk[0]; xmax = Math.max(xmax, tk[tk.length - 1]); x = function (v) { return L + (W - L - R) * (Math.log10(Math.max(v, xmin)) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)); }; }
      else if (o.ticks && o.ticks.step) { var st = o.ticks.step, k = Math.max(1, Math.ceil((xmax - xmin) / st / (narrow ? 4 : 8))); xmax = Math.ceil(xmax / (st * k)) * st * k; tk = []; for (var v = xmin; v <= xmax + 1e-9; v += st * k) tk.push(v); x = function (v) { return L + (W - L - R) * (v - xmin) / (xmax - xmin); }; }
      else { tk = ticks(xmin, xmax, narrow ? 4 : 6); xmax = tk[tk.length - 1]; x = function (v) { return L + (W - L - R) * (v - xmin) / (xmax - xmin); }; }
      var s = svgOpen(W, H, o.title);
      tk.forEach(function (t) { s += '<line x1="' + x(t) + '" x2="' + x(t) + '" y1="' + top + '" y2="' + (H - B) + '" stroke="var(--grid)"/><text x="' + x(t) + '" y="' + (H - B + 16) + '" text-anchor="middle">' + fmt(t) + "</text>"; });
      if (o.xLabel) s += '<text x="' + ((L + W - R) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.xLabel) + (o.log ? " (log scale)" : "") + "</text>";
      (o.refs || []).forEach(function (rf) { s += '<line x1="' + x(rf.x) + '" x2="' + x(rf.x) + '" y1="' + top + '" y2="' + (H - B) + '" stroke="var(--ink-3)"/><text x="' + (x(rf.x) + 4) + '" y="' + (top + 10) + '">' + esc(rf.label) + "</text>"; });
      o.rows.forEach(function (r, i) {
        var y0 = top + i * rowH;
        s += narrow ? '<text class="op" x="0" y="' + (y0 + 12) + '">' + esc(r.label) + "</text>" : '<text class="op" x="0" y="' + (y0 + lab + series.length * laneH / 2 + 4) + '">' + esc(r.label) + "</text>";
        series.forEach(function (se, j) {
          var p = r.series[se.key]; if (!p) return; var y = y0 + lab + j * laneH + laneH / 2;
          s += '<g data-tip="' + esc("<b>" + se.name + "</b> · " + r.label + "<br>" + (p.tip || (fmt(p.mid) + " [" + fmt(p.lo) + ", " + fmt(p.hi) + "]" + (o.unit ? " " + o.unit : ""))) ) + '">' +
            '<rect x="' + (x(p.lo) - 4) + '" y="' + (y - 7) + '" width="' + (x(p.hi) - x(p.lo) + 8) + '" height="14" fill="transparent"/>' +
            '<line x1="' + x(p.lo) + '" x2="' + x(p.hi) + '" y1="' + y + '" y2="' + y + '" stroke="' + se.color + '" stroke-width="2" stroke-linecap="round"/>' +
            '<circle cx="' + x(p.mid) + '" cy="' + y + '" r="5" fill="' + se.color + '" stroke="var(--surface-1)" stroke-width="2"/>' +
            (p.mark != null ? '<circle cx="' + x(p.mark) + '" cy="' + y + '" r="5" fill="var(--surface-1)" stroke="' + se.color + '" stroke-width="2"/>' : "") + "</g>";
        });
      });
      el.innerHTML = legend(series) + s + "</svg>";
      bindTip(el, el.querySelector("svg.chart"));
    });
  };

  /* ---------- line chart with optional band ---------- */
  C.line = function (target, o) {
    var fmt = o.format || deffmt, xfmt = o.xFormat || deffmt;
    return mount(target, function (el, W) {
      var narrow = W < 480, H = Math.round(o.height || Math.max(220, Math.min(W * (narrow ? 0.9 : 0.5), 420))), L = 48, R = o.endLabels === false ? 16 : (narrow ? 16 : 96), T = 10, B = 40;
      var xs = o.x, xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs), ymin = Infinity, ymax = -Infinity;
      o.series.forEach(function (s) { s.y.forEach(function (v, i) { if (v == null) return; var a = s.lo ? s.lo[i] : v, b = s.hi ? s.hi[i] : v; ymin = Math.min(ymin, a, v); ymax = Math.max(ymax, b, v); }); });
      if (o.zero !== false && !o.log) ymin = Math.min(0, ymin);
      var yt = o.log ? logTicks(ymin, ymax) : ticks(o.ymin != null ? o.ymin : ymin, o.ymax != null ? o.ymax : ymax, narrow ? 4 : 5), y0 = o.log ? yt[0] : yt[0], y1 = Math.max(yt[yt.length - 1], ymax);
      var xt = o.xTicks || ticks(xmin, xmax, narrow ? 3 : 6).filter(function (t) { return t >= xmin && t <= xmax; });
      var X = function (v) { return L + (W - L - R) * (v - xmin) / ((xmax - xmin) || 1); };
      var Y = o.log ? function (v) { return T + (H - T - B) * (1 - (Math.log10(Math.max(v, y0)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))); } : function (v) { return T + (H - T - B) * (1 - (v - y0) / ((y1 - y0) || 1)); };
      var s = svgOpen(W, H, o.title);
      yt.forEach(function (t) { s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(t) + '" y2="' + Y(t) + '" stroke="var(--grid)"/><text x="' + (L - 8) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + fmt(t) + "</text>"; });
      xt.forEach(function (t) { s += '<text x="' + X(t) + '" y="' + (H - B + 18) + '" text-anchor="middle">' + xfmt(t) + "</text>"; });
      if (o.xLabel) s += '<text x="' + ((L + W - R) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.xLabel) + "</text>";
      (o.refs || []).forEach(function (rf) { s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(rf.y) + '" y2="' + Y(rf.y) + '" stroke="var(--ink-3)"/><text x="' + (W - R) + '" y="' + (Y(rf.y) - 6) + '" text-anchor="end">' + esc(rf.label) + "</text>"; });
      o.series.forEach(function (se) {
        if (se.lo && se.hi) { var up = [], dn = []; xs.forEach(function (xv, i) { if (se.lo[i] == null) return; up.push(X(xv) + "," + Y(se.hi[i])); dn.unshift(X(xv) + "," + Y(se.lo[i])); }); s += '<polygon points="' + up.concat(dn).join(" ") + '" fill="' + se.color + '" opacity=".12"/>'; }
        var d = "", pen = false; xs.forEach(function (xv, i) { var v = se.y[i]; if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + X(xv) + "," + Y(v); pen = true; });
        s += '<path d="' + d + '" fill="none" stroke="' + se.color + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"' + (se.dash ? ' stroke-dasharray="5 4"' : "") + "/>";
        var li = se.y.length - 1; while (li > 0 && se.y[li] == null) li--;
        if (o.endLabels !== false && !narrow && o.series.length <= 4) s += '<text class="lbl" x="' + (X(xs[li]) + 8) + '" y="' + (Y(se.y[li]) + 4) + '">' + esc(se.name) + "</text>";
      });
      /* crosshair hit layer */
      s += '<rect class="hit" x="' + L + '" y="' + T + '" width="' + (W - L - R) + '" height="' + (H - T - B) + '" fill="transparent"/><line class="xh" x1="0" x2="0" y1="' + T + '" y2="' + (H - B) + '" stroke="var(--ink-3)" visibility="hidden"/>';
      el.innerHTML = legend(o.series) + s + "</svg>";
      var svg = el.querySelector("svg.chart"), hit = svg.querySelector(".hit"), xh = svg.querySelector(".xh"), tp = tip(el);
      function at(cx) { var r = svg.getBoundingClientRect(), px = (cx - r.left) * W / r.width, best = 0; xs.forEach(function (xv, i) { if (Math.abs(X(xv) - px) < Math.abs(X(xs[best]) - px)) best = i; }); return best; }
      hit.addEventListener("mousemove", function (e) { var i = at(e.clientX), r = el.getBoundingClientRect(), sr = svg.getBoundingClientRect(); xh.setAttribute("x1", X(xs[i])); xh.setAttribute("x2", X(xs[i])); xh.setAttribute("visibility", "visible");
        tp.show("<b>" + esc(xfmt(xs[i])) + "</b>" + o.series.map(function (se) { return se.y[i] == null ? "" : "<br>" + esc(se.name) + ": " + fmt(se.y[i]) + (o.yUnit ? " " + esc(o.yUnit) : ""); }).join(""), (X(xs[i]) * sr.width / W) + sr.left - r.left, sr.top - r.top + 20); });
      hit.addEventListener("mouseleave", function () { xh.setAttribute("visibility", "hidden"); tp.hide(); });
      if (o.yLabel) { var cap = document.createElement("div"); cap.style.cssText = "font-size:12px;color:var(--ink-3);margin-bottom:4px"; cap.textContent = o.yLabel; el.insertBefore(cap, svg); }
    });
  };

  /* ---------- table twin ---------- */
  C.table = function (target, o) {
    if (typeof target === "string") target = document.querySelector(target);
    if (!target) return;
    var h = '<div class="tablewrap" tabindex="0" role="region" aria-label="' + esc(o.caption || "Data table") + '"><table>' + (o.caption ? "<caption>" + esc(o.caption) + "</caption>" : "") + "<thead><tr>";
    o.columns.forEach(function (c, i) { h += '<th scope="col"' + (c.align === "left" && i ? ' style="text-align:left"' : "") + ">" + esc(c.label) + "</th>"; });
    h += "</tr></thead><tbody>";
    o.rows.forEach(function (r) { h += "<tr>" + o.columns.map(function (c, i) { var v = r[c.key], t = c.format ? c.format(v, r) : v == null ? "—" : v; return i === 0 ? '<th scope="row">' + esc(t) + "</th>" : "<td" + (c.align === "left" ? ' style="text-align:left"' : "") + ">" + esc(t) + "</td>"; }).join("") + "</tr>"; });
    target.innerHTML = h + "</tbody></table></div>";
    target.classList.add("rk-table");
  };
})();
