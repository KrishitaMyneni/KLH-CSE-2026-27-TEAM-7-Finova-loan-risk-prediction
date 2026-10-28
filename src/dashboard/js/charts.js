/*
 * charts.js — portfolio charts on the bundled Chart.js (v4, offline).
 *
 * Rule for encodings: risk colours (approve / review / decline) are only used
 * where the value actually is a risk band or decision. Anything else (exposure,
 * expected loss, score distribution) uses the neutral slate ramp.
 */
window.CRC = window.CRC || {};
(function (CRC) {
  "use strict";

  var dom = CRC.dom;
  var fmt = CRC.fmt;

  var instances = {};

  function theme() {
    return {
      text: dom.token("--text-secondary"),
      muted: dom.token("--text-tertiary"),
      grid: dom.token("--border-light"),
      surface: dom.token("--bg-surface"),
      neutralStrong: dom.token("--text-secondary"),
      neutralSoft: dom.token("--border-control"),
      approve: dom.token("--color-approve"),
      review: dom.token("--color-review"),
      decline: dom.token("--color-decline"),
      ineligible: dom.token("--color-ineligible"),
      approveBg: dom.token("--color-approve-bg"),
      reviewBg: dom.token("--color-review-bg"),
      declineBg: dom.token("--color-decline-bg"),
      approveBorder: dom.token("--color-approve-border"),
      reviewBorder: dom.token("--color-review-border"),
      declineBorder: dom.token("--color-decline-border")
    };
  }

  function baseOptions(t) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 180 },
      layout: { padding: { top: 4, right: 4 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: t.surface,
          borderColor: t.grid,
          borderWidth: 1,
          titleColor: t.text,
          bodyColor: t.text,
          padding: 10,
          cornerRadius: 6,
          displayColors: true,
          boxWidth: 8,
          boxHeight: 8,
          usePointStyle: true,
          titleFont: { family: "IBM Plex Sans", size: 12, weight: "600" },
          bodyFont: { family: "IBM Plex Sans", size: 12 }
        }
      }
    };
  }

  /**
   * Canvas cannot resolve color-mix(), so blend a token toward the surface
   * colour here. weight = share of the token colour (0-1).
   */
  function blend(color, base, weight) {
    function rgb(value) {
      var hex = String(value).trim().replace("#", "");
      if (!/^[0-9a-fA-F]{3}$|^[0-9a-fA-F]{6}$/.test(hex)) return null;
      if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
      var n = parseInt(hex, 16);
      return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
    }
    var a = rgb(color);
    var b = rgb(base);
    if (!a || !b) return color;
    return "rgb(" +
      Math.round(a.r * weight + b.r * (1 - weight)) + ", " +
      Math.round(a.g * weight + b.g * (1 - weight)) + ", " +
      Math.round(a.b * weight + b.b * (1 - weight)) + ")";
  }

  function axisTitle(text, t) {
    return {
      display: true,
      text: text,
      color: t.muted,
      font: { family: "IBM Plex Sans", size: 11 },
      padding: { top: 6 }
    };
  }

  function tickStyle(t) {
    return {
      color: t.muted,
      font: { family: "IBM Plex Sans", size: 11 }
    };
  }

  function mount(canvasId, config) {
    var canvas = dom.qs("#" + canvasId);
    if (!canvas || typeof window.Chart === "undefined") return null;
    if (instances[canvasId]) instances[canvasId].destroy();
    instances[canvasId] = new window.Chart(canvas, config);
    return instances[canvasId];
  }

  function destroyAll() {
    Object.keys(instances).forEach(function (id) {
      instances[id].destroy();
      delete instances[id];
    });
  }

  function hiddenTable(hostId, caption, headers, rows) {
    var host = dom.qs("#" + hostId);
    if (!host) return;
    host.innerHTML = "<table><caption>" + CRC.esc(caption) + "</caption><thead><tr>" +
      headers.map(function (h) { return "<th>" + CRC.esc(h) + "</th>"; }).join("") +
      "</tr></thead><tbody>" +
      rows.map(function (row) {
        return "<tr>" + row.map(function (cell) { return "<td>" + CRC.esc(cell) + "</td>"; }).join("") + "</tr>";
      }).join("") + "</tbody></table>";
  }

  /* ----------------------------------------------------------------------
   * 1. Decision mix — semantic colours, because these are decisions.
   * -------------------------------------------------------------------- */
  function renderDecision(data) {
    var t = theme();
    var s = data.summary;
    var labels = ["Approve", "Review", "Decline", "Ineligible"];
    var values = [s.approved_count, s.review_count, s.declined_count, s.ineligible_count];
    var colors = [t.approve, t.review, t.decline, t.ineligible].map(function (color) {
      return blend(color, t.surface, 0.85);
    });

    var options = baseOptions(t);
    options.indexAxis = "y";
    options.scales = {
      x: {
        beginAtZero: true,
        grid: { color: t.grid, drawTicks: false },
        border: { display: false },
        ticks: Object.assign(tickStyle(t), { precision: 0 }),
        title: axisTitle("Records", t)
      },
      y: {
        grid: { display: false },
        border: { display: false },
        ticks: Object.assign(tickStyle(t), { font: { family: "IBM Plex Sans", size: 12 } })
      }
    };
    options.plugins.tooltip.callbacks = {
      label: function (ctx) {
        var total = values.reduce(function (a, b) { return a + b; }, 0);
        var share = total ? ((ctx.raw / total) * 100).toFixed(1) : "0.0";
        return ctx.label + ": " + fmt.count(ctx.raw) + " (" + share + "%)";
      }
    };

    mount("chart-decision", {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderRadius: 2,
          barThickness: 22
        }]
      },
      options: options
    });

    hiddenTable("table-decision", "Decision mix", ["Decision", "Records"],
      labels.map(function (label, i) { return [label, fmt.count(values[i])]; }));
  }

  /* ----------------------------------------------------------------------
   * 2. PD distribution — bars coloured by the policy zone they fall in.
   * -------------------------------------------------------------------- */
  var PD_BINS = [
    { label: "0–5%", min: 0, max: 0.05 },
    { label: "5–10%", min: 0.05, max: 0.10 },
    { label: "10–15%", min: 0.10, max: 0.15 },
    { label: "15–20%", min: 0.15, max: 0.20 },
    { label: "20–25%", min: 0.20, max: 0.25 },
    { label: "25–30%", min: 0.25, max: 0.30 },
    { label: "30%+", min: 0.30, max: Infinity }
  ];

  function renderPdHistogram(data, eligible) {
    var t = theme();
    var counts = PD_BINS.map(function () { return 0; });

    eligible.forEach(function (app) {
      var pd = app.default_probability;
      if (pd === null || pd === undefined) return;
      for (var i = 0; i < PD_BINS.length; i++) {
        if (pd >= PD_BINS[i].min && pd < PD_BINS[i].max) {
          counts[i] += 1;
          break;
        }
      }
    });

    var zoneSource = PD_BINS.map(function (bin, i) {
      if (i < 2) return t.approve;
      if (i < 4) return t.review;
      return t.decline;
    });
    var zoneColors = zoneSource.map(function (color) {
      return blend(color, t.surface, 0.45);
    });
    var zoneBorders = zoneSource.map(function (color) {
      return blend(color, t.surface, 0.8);
    });

    var options = baseOptions(t);
    options.scales = {
      x: {
        grid: { display: false },
        border: { color: t.grid },
        ticks: tickStyle(t),
        title: axisTitle("Predicted probability of default", t)
      },
      y: {
        beginAtZero: true,
        grid: { color: t.grid, drawTicks: false },
        border: { display: false },
        ticks: Object.assign(tickStyle(t), { precision: 0 }),
        title: axisTitle("Applicants", t)
      }
    };
    options.plugins.tooltip.callbacks = {
      label: function (ctx) { return fmt.count(ctx.raw) + " applicants"; },
      afterLabel: function (ctx) {
        if (ctx.dataIndex < 2) return "Approve zone (10% or less)";
        if (ctx.dataIndex < 4) return "Review zone (10% to 20%)";
        return "Decline zone (above 20%)";
      }
    };

    mount("chart-pd", {
      type: "bar",
      data: {
        labels: PD_BINS.map(function (b) { return b.label; }),
        datasets: [{
          data: counts,
          backgroundColor: zoneColors,
          borderColor: zoneBorders,
          borderWidth: 1,
          borderRadius: 2
        }]
      },
      options: options
    });

    hiddenTable("table-pd", "Probability of default by band", ["PD band", "Applicants"],
      PD_BINS.map(function (b, i) { return [b.label, fmt.count(counts[i])]; }));
  }

  /* ----------------------------------------------------------------------
   * 3. Exposure vs expected loss. Exposure is faint context; expected loss is
   *    the figure that matters, coloured by risk band because the x-axis is
   *    the risk band.
   * -------------------------------------------------------------------- */
  function renderExposure(data, eligible) {
    var t = theme();
    var bands = ["Low Risk", "Medium Risk", "High Risk"];
    var exposure = [0, 0, 0];
    var loss = [0, 0, 0];

    eligible.forEach(function (app) {
      var i = bands.indexOf(app.risk_band);
      if (i === -1) return;
      exposure[i] += Number(app.loan_amnt) || 0;
      loss[i] += Number(app.expected_loss) || 0;
    });

    var options = baseOptions(t);
    options.plugins.legend = {
      display: true,
      position: "bottom",
      align: "start",
      labels: {
        color: t.text,
        boxWidth: 8,
        boxHeight: 8,
        usePointStyle: true,
        pointStyle: "rectRounded",
        padding: 14,
        font: { family: "IBM Plex Sans", size: 11 }
      }
    };
    options.scales = {
      x: {
        grid: { display: false },
        border: { color: t.grid },
        ticks: tickStyle(t)
      },
      y: {
        beginAtZero: true,
        grid: { color: t.grid, drawTicks: false },
        border: { display: false },
        ticks: Object.assign(tickStyle(t), {
          callback: function (value) { return fmt.moneyCompact(value); }
        }),
        title: axisTitle("USD", t)
      }
    };
    options.plugins.tooltip.callbacks = {
      label: function (ctx) { return ctx.dataset.label + ": " + fmt.money(ctx.raw); },
      afterBody: function (items) {
        var i = items[0].dataIndex;
        return exposure[i] > 0 ? "Expected-loss rate: " + ((loss[i] / exposure[i]) * 100).toFixed(1) + "%" : "";
      }
    };
    var bandColors = [t.approve, t.review, t.decline];
    // One swatch per series: band colours would otherwise show only the first band
    options.plugins.legend.labels.generateLabels = function (chart) {
      return Chart.defaults.plugins.legend.labels.generateLabels(chart).map(function (item) {
        if (item.datasetIndex === 1) {
          item.fillStyle = t.neutralStrong;
          item.strokeStyle = t.neutralStrong;
        }
        return item;
      });
    };

    mount("chart-exposure", {
      type: "bar",
      data: {
        labels: bands,
        datasets: [
          {
            label: "Requested exposure",
            data: exposure,
            backgroundColor: blend(t.neutralSoft, t.surface, 0.45),
            borderRadius: 3
          },
          {
            label: "Expected loss",
            data: loss,
            backgroundColor: bandColors,
            borderRadius: 3
          }
        ]
      },
      options: options
    });

    hiddenTable("table-exposure", "Exposure and expected loss by risk band",
      ["Risk band", "Requested exposure", "Expected loss"],
      bands.map(function (band, i) {
        return [band, fmt.money(exposure[i]), fmt.money(loss[i])];
      }));
  }

  /* ----------------------------------------------------------------------
   * 4. Model score distribution — one neutral series, no traffic lights.
   * -------------------------------------------------------------------- */
  var SCORE_BINS = [
    { label: "Below 620", min: -Infinity, max: 620 },
    { label: "620–659", min: 620, max: 660 },
    { label: "660–699", min: 660, max: 700 },
    { label: "700–739", min: 700, max: 740 },
    { label: "740–779", min: 740, max: 780 },
    { label: "780–850", min: 780, max: Infinity }
  ];

  function renderScoreDistribution(data, eligible) {
    var t = theme();
    var counts = SCORE_BINS.map(function () { return 0; });

    eligible.forEach(function (app) {
      var score = app.credit_score;
      if (score === null || score === undefined) return;
      for (var i = 0; i < SCORE_BINS.length; i++) {
        if (score >= SCORE_BINS[i].min && score < SCORE_BINS[i].max) {
          counts[i] += 1;
          break;
        }
      }
    });

    var options = baseOptions(t);
    options.scales = {
      x: {
        grid: { display: false },
        border: { color: t.grid },
        ticks: tickStyle(t),
        title: axisTitle("Model score, 300 to 850", t)
      },
      y: {
        beginAtZero: true,
        grid: { color: t.grid, drawTicks: false },
        border: { display: false },
        ticks: Object.assign(tickStyle(t), { precision: 0 }),
        title: axisTitle("Applicants", t)
      }
    };
    options.plugins.tooltip.callbacks = {
      label: function (ctx) { return fmt.count(ctx.raw) + " applicants"; }
    };

    mount("chart-score", {
      type: "bar",
      data: {
        labels: SCORE_BINS.map(function (b) { return b.label; }),
        datasets: [{
          data: counts,
          backgroundColor: t.neutralSoft,
          borderRadius: 2
        }]
      },
      options: options
    });

    hiddenTable("table-score", "Model score distribution", ["Score range", "Applicants"],
      SCORE_BINS.map(function (b, i) { return [b.label, fmt.count(counts[i])]; }));
  }

  function renderAll(data) {
    if (!data || typeof window.Chart === "undefined") return;
    var eligible = (data.applicants || []).filter(function (app) {
      return app.decision !== "INELIGIBLE" && app.default_probability !== null;
    });

    renderDecision(data);
    renderPdHistogram(data, eligible);
    renderExposure(data, eligible);
    renderScoreDistribution(data, eligible);
  }

  CRC.charts = {
    renderAll: renderAll,
    destroyAll: destroyAll
  };
})(window.CRC);
