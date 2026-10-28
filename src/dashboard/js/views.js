/*
 * views.js — rendering for the assessment panel, batch results and the
 * applicant drawer. Pure render functions over the API payload; the only
 * state they mutate is CRC.state.rows / pagination output.
 */
window.CRC = window.CRC || {};
(function (CRC) {
  "use strict";

  var fmt = CRC.fmt;
  var dom = CRC.dom;
  var esc = CRC.esc;
  var icon = CRC.icon;
  var state = CRC.state;

  var TABLE_COLUMNS = [
    { key: "id", label: "Applicant", sortable: true, num: false },
    { key: "loan_amnt", label: "Loan amount", sortable: true, num: true },
    { key: "default_probability", label: "Prob. of default", sortable: true, num: true },
    { key: "credit_score", label: "Model score", sortable: true, num: true },
    { key: "expected_loss", label: "Expected loss", sortable: true, num: true },
    { key: "decision", label: "Decision", sortable: false, num: false }
  ];

  /* ----------------------------------------------------------------------
   * Shared bits
   * -------------------------------------------------------------------- */
  function noticeHtml(kind, message) {
    return '<span class="notice__icon">' + icon(kind === "warn" ? "alert" : "alert", 15) + "</span>" +
      "<span>" + esc(message) + "</span>";
  }

  function stateHtml(title, body, actionLabel, action) {
    return '<div class="state state--center">' +
      '<p class="state__title">' + esc(title) + "</p>" +
      (body ? '<p class="state__body">' + esc(body) + "</p>" : "") +
      (actionLabel ? '<button class="btn btn--secondary btn--sm" type="button" data-action="' + esc(action) + '">' +
        esc(actionLabel) + "</button>" : "") +
      "</div>";
  }

  function plural(n, word) {
    return fmt.count(n) + " " + word + (Number(n) === 1 ? "" : "s");
  }

  function maybeMoney(raw) {
    if (raw === null || raw === undefined || raw === "" || raw === "N/A" || raw === "nan") {
      return fmt.EM_DASH;
    }
    var text = String(raw).trim();
    if (/^-?\d+(\.\d+)?$/.test(text)) return fmt.money(Number(text));
    return esc(text);
  }

  function maybePct(raw) {
    if (raw === null || raw === undefined || raw === "" || raw === "N/A" || raw === "nan") {
      return fmt.EM_DASH;
    }
    var text = String(raw).trim();
    if (/^-?\d+(\.\d+)?$/.test(text)) return fmt.pctRaw(Number(text));
    return esc(text);
  }

  function maybeText(raw) {
    if (raw === null || raw === undefined || raw === "" || raw === "N/A" || raw === "nan") {
      return fmt.EM_DASH;
    }
    return esc(raw);
  }

  /* ----------------------------------------------------------------------
   * Single applicant assessment
   * -------------------------------------------------------------------- */
  function scoreCell(label, valueHtml, sub, options) {
    var opts = options || {};
    var classes = "score-panel__value";
    if (opts.large) classes += " score-panel__value--lg";
    if (opts.muted) classes += " score-panel__value--muted";
    return '<div class="score-panel__cell">' +
      '<div class="score-panel__label">' + esc(label) + "</div>" +
      '<div class="' + classes + '">' + valueHtml + "</div>" +
      (sub ? '<div class="score-panel__sub">' + sub + "</div>" : "") +
      "</div>";
  }

  /** The decision, in words a loan officer acts on, as the page headline. */
  function decisionHero(app, data, scored) {
    var low = Math.round((Number(data.tau_low) || 0.10) * 100);
    var high = Math.round((Number(data.tau_high) || 0.20) * 100);
    var pd = scored ? fmt.pct(app.default_probability) : "";
    var reasons = {
      APPROVE: pd + " chance of default, inside the " + low + "% approval limit.",
      REVIEW: pd + " chance of default, between the " + low + "% and " + high + "% limits. Send to an underwriter.",
      DECLINE: pd + " chance of default, above the " + high + "% limit.",
      INELIGIBLE: "Not scored. Only 36-month loans are inside the model's validated scope."
    };
    var kind = { APPROVE: "approve", REVIEW: "review", DECLINE: "decline" }[app.decision] || "neutral";
    return '<div class="decision-hero decision-hero--' + kind + '">' +
      '<p class="decision-hero__word">' + esc(fmt.decisionLabel(app.decision)) + "</p>" +
      '<p class="decision-hero__reason">' + esc(reasons[app.decision] || "") + "</p>" +
      "</div>";
  }

  /** PD against the live policy cutoffs returned by the API (tau_low/tau_high). */
  function probabilityScale(pd, tauLow, tauHigh) {
    var low = Number(tauLow) || 0.10;
    var high = Number(tauHigh) || 0.20;
    var max = high * 1.5;
    var pos = dom.clamp((pd / max) * 100, 0, 100);
    var lowPct = Math.round(low * 100);
    var highPct = Math.round(high * 100);
    var maxPct = Math.round(max * 100);

    return '<div class="scale" role="img" aria-label="Probability of default ' + fmt.pct(pd) +
      ' against policy cutoffs of ' + lowPct + '% and ' + highPct + '%">' +
      '<div class="scale__track">' +
      '<div class="scale__zone scale__zone--low"></div>' +
      '<div class="scale__zone scale__zone--med"></div>' +
      '<div class="scale__zone scale__zone--high"></div>' +
      '<div class="scale__marker" style="left:' + pos.toFixed(2) + '%"></div>' +
      "</div>" +
      '<div class="scale__ticks"><span>0%</span><span>' + lowPct + "%</span><span>" + highPct +
      "%</span><span>" + maxPct + "%+</span></div>" +
      '<div class="scale__legend">' +
      '<span><span class="scale__swatch scale__swatch--low"></span>Approve, ' + lowPct + "% or less</span>" +
      '<span><span class="scale__swatch scale__swatch--med"></span>Review, ' + lowPct + "% to " + highPct + "%</span>" +
      '<span><span class="scale__swatch scale__swatch--high"></span>Decline, above ' + highPct + "%</span>" +
      "</div></div>";
  }

  function driverTable(codes) {
    if (!codes || !codes.length) {
      return stateHtml("No drivers returned", "This record was not scored, so no feature contributions exist.");
    }

    var maxAbs = codes.reduce(function (acc, rc) {
      return Math.max(acc, Math.abs(Number(rc.shap_value) || 0));
    }, 0) || 1;

    var rows = codes.map(function (rc) {
      var value = Number(rc.shap_value) || 0;
      var up = value > 0;
      var width = (Math.abs(value) / maxAbs) * 50;
      var label = rc.label || fmt.humanKey(rc.feature);
      var shown = rc.value !== undefined && rc.value !== null && String(rc.value) !== "N/A"
        ? String(rc.value) : "";

      var bar = up
        ? '<span class="driver-bar__fill" style="left:50%;width:' + width.toFixed(2) + '%"></span>'
        : '<span class="driver-bar__fill driver-bar__fill--down" style="right:50%;width:' + width.toFixed(2) + '%"></span>';

      return "<tr>" +
        '<td><span class="driver-name" title="' + esc(label) + '">' + esc(label) + "</span></td>" +
        '<td class="num">' + (shown ? esc(shown) : fmt.EM_DASH) + "</td>" +
        '<td class="col-direction"><span class="direction">' + icon(up ? "arrow-up" : "arrow-down", 13) +
        (up ? "Increases risk" : "Reduces risk") + "</span></td>" +
        '<td class="num col-contrib">' + (up ? "+" : "") + value.toFixed(3) + "</td>" +
        '<td class="col-impact"><span class="driver-bar">' +
        '<span class="driver-bar__center"></span>' + bar + "</span></td>" +
        "</tr>";
    }).join("");

    return '<table class="data-table"><thead><tr>' +
      "<th>Driver</th><th class=\"num\">Input</th><th class=\"col-direction\">Direction</th>" +
      "<th class=\"num col-contrib\">Contribution</th><th>Impact</th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table>";
  }

  function comparisonTable(app, benchmarks) {
    if (!benchmarks) return "";
    var rows = [
      { label: "Bureau FICO", applicant: fmt.num(app.fico, 0), base: benchmarks.fico_avg },
      { label: "DTI ratio", applicant: fmt.pctRaw(app.dti, 1), base: benchmarks.dti === undefined ? null : fmt.pctRaw(benchmarks.dti, 1) },
      { label: "Annual income", applicant: fmt.money(app.annual_inc), base: benchmarks.annual_inc === undefined ? null : fmt.money(benchmarks.annual_inc) },
      { label: "Revolving utilization", applicant: fmt.pctRaw(app.raw_features ? app.raw_features.revol_util : null, 1), base: benchmarks.revol_util === undefined ? null : fmt.pctRaw(benchmarks.revol_util, 1) }
    ];

    return '<table class="data-table"><thead><tr>' +
      "<th>Measure</th><th class=\"num\">Applicant</th><th class=\"num\">Portfolio average</th>" +
      "</tr></thead><tbody>" +
      rows.map(function (row) {
        return '<tr class="compare-row"><td>' + esc(row.label) + "</td>" +
          '<td class="num">' + esc(row.applicant) + "</td>" +
          '<td class="num">' + esc(row.base === null ? fmt.EM_DASH : row.base) + "</td></tr>";
      }).join("") +
      "</tbody></table>";
  }

  function renderAssessment(data, app) {
    var host = dom.qs("#view-assessment");
    if (!host || !app) return;

    var scored = app.decision !== "INELIGIBLE" && app.default_probability !== null;
    var html = "";

    html += '<div class="assessment-head">' +
      '<div class="assessment-head__identity">' +
      decisionHero(app, data, scored) +
      '<div class="assessment-head__meta">' + esc(fmt.termLabel(app.term)) +
      " · " + esc(fmt.humanPurpose(app.purpose)) +
      " · " + esc(fmt.money(app.loan_amnt)) + " requested</div>" +
      "</div>" +
      '<div class="result-head__actions">' +
      '<button class="btn btn--secondary btn--sm" type="button" data-action="edit-inputs">Edit inputs</button>' +
      '<button class="btn btn--ghost btn--sm" type="button" data-action="new-assessment">New applicant</button>' +
      "</div></div>";

    html += '<div class="score-panel"><div class="score-panel__grid">' +
      scoreCell("Probability of default",
        scored ? esc(fmt.pct(app.default_probability)) : fmt.NOT_SCORED,
        scored ? "Calibrated, so it reads as a real default rate" : "Policy exclusion",
        { muted: !scored }) +
      scoreCell("Model score",
        scored ? esc(app.credit_score) + '<span class="score-panel__suffix">/ 850</span>' : fmt.NOT_SCORED,
        scored ? "Scorecard scale, 300 to 850" : "",
        { muted: !scored }) +
      '<div class="score-panel__cell">' +
      '<div class="score-panel__label">Risk band</div>' +
      '<div class="score-panel__band">' + fmt.bandDot(app.risk_band, app.decision) + "</div>" +
      "</div>" +
      scoreCell("Expected loss",
        scored ? esc(fmt.money(app.expected_loss)) : fmt.NOT_SCORED,
        scored ? "PD × LGD × loan amount" : "",
        { muted: !scored }) +
      "</div>" +
      (scored ? probabilityScale(app.default_probability, data.tau_low, data.tau_high) : "") +
      "</div>";

    html += '<div class="block"><div class="block__head">' +
      '<h2 class="block__title">Key drivers</h2>' +
      '<span class="block__note">Log-odds contribution, top 4</span>' +
      "</div>" + driverTable(app.reason_codes) + "</div>";

    // The decision headline already states the rationale, so it is not repeated here
    if (app.warning) {
      html += '<div class="block"><div class="notice notice--warn">' +
        noticeHtml("warn", app.warning) + "</div></div>";
    }

    html += '<div class="block"><div class="block__head">' +
      '<h2 class="block__title">Against portfolio averages</h2>' +
      '<span class="block__note">Training-window benchmarks</span></div>' +
      comparisonTable(app, data.benchmarks) + "</div>";

    if (app.imputed_labels && app.imputed_labels.length) {
      html += '<div class="block"><details class="disclosure">' +
        "<summary>" + icon("chevron-right", 13) +
        esc(plural(app.imputed_labels.length, "field") + " filled in automatically") + "</summary>" +
        '<div class="disclosure__body">These inputs were blank, so the model used the training-data median: ' +
        esc(app.imputed_labels.join(", ")) + ".</div></details></div>";
    }

    html += '<div class="block"><details class="disclosure">' +
      "<summary>" + icon("chevron-right", 13) + "Submitted attributes</summary>" +
      '<div class="disclosure__body">' + attributesGrid(app.raw_features) + "</div></details></div>";

    host.innerHTML = html;
  }

  function attributesGrid(raw) {
    if (!raw || !Object.keys(raw).length) {
      return '<p class="state__body">No attributes returned.</p>';
    }
    var order = Object.keys(raw).sort(function (a, b) {
      return fmt.humanKey(a).localeCompare(fmt.humanKey(b));
    });
    return '<div class="kv-grid">' + order.map(function (key) {
      return '<div class="kv"><div class="kv__label">' + esc(fmt.humanKey(key)) + "</div>" +
        '<div class="kv__value">' + esc(raw[key]) + "</div></div>";
    }).join("") + "</div>";
  }

  /* ----------------------------------------------------------------------
   * Batch results
   * -------------------------------------------------------------------- */
  function renderResultsMeta(data) {
    var el = dom.qs("#results-meta");
    if (!el) return;
    var s = data.summary;
    el.innerHTML =
      "<strong>" + fmt.count(s.total_applicants) + "</strong> records · " +
      "<strong>" + fmt.count(s.eligible_count) + "</strong> scored · " +
      "<strong>" + fmt.count(s.ineligible_count) + "</strong> ineligible · " +
      "<strong>" + fmt.count(s.could_not_score_count) + "</strong> invalid";
  }

  function renderToolbarCounts(data) {
    var host = dom.qs("#decision-filters");
    if (!host) return;
    var s = data.summary;
    var options = [
      { value: "ALL", label: "All", count: (data.applicants || []).length },
      { value: "APPROVE", label: "Approve", count: s.approved_count },
      { value: "REVIEW", label: "Review", count: s.review_count },
      { value: "DECLINE", label: "Decline", count: s.declined_count },
      { value: "INELIGIBLE", label: "Ineligible", count: s.ineligible_count }
    ];

    host.innerHTML = options.map(function (opt) {
      var pressed = state.filters.decision === opt.value;
      return '<button class="filter-btn" type="button" data-action="filter-decision" data-decision="' +
        opt.value + '" aria-pressed="' + (pressed ? "true" : "false") + '">' +
        esc(opt.label) + '<span class="filter-btn__count">' + fmt.count(opt.count) + "</span></button>";
    }).join("");
  }

  function sortIcon(field) {
    if (state.sort.field !== field) return icon("chevs-up-down", 12);
    return icon(state.sort.dir === "asc" ? "arrow-up" : "arrow-down", 12);
  }

  function renderApplicantTable() {
    var table = dom.qs("#applicants-table");
    if (!table) return;
    var thead = table.querySelector("thead");
    var tbody = table.querySelector("tbody");

    thead.innerHTML = "<tr>" + TABLE_COLUMNS.map(function (col) {
      var sortAttr = col.key === state.sort.field
        ? (state.sort.dir === "asc" ? ' aria-sort="ascending"' : ' aria-sort="descending"')
        : "";
      var content = col.sortable
        ? '<button class="th-sort" type="button" data-action="sort" data-field="' + col.key + '">' +
          esc(col.label) + sortIcon(col.key) + "</button>"
        : esc(col.label);
      return "<th" + (col.num ? ' class="num"' : "") + sortAttr + ">" + content + "</th>";
    }).join("") + '<th class="cell-action"><span class="visually-hidden">Open</span></th></tr>';

    var total = state.rows.length;
    var start = (state.page - 1) * state.pageSize;
    var rows = state.rows.slice(start, start + state.pageSize);

    if (!rows.length) {
      var filtered = state.filters.q || state.filters.decision !== "ALL" || state.filters.band !== "ALL";
      tbody.innerHTML = '<tr><td class="cell-empty" colspan="' + (TABLE_COLUMNS.length + 1) + '">' +
        (filtered
          ? stateHtml("No rows match these filters", "Adjust the search, decision or risk band.", "Clear filters", "clear-filters")
          : stateHtml("No applicants in this batch", "The uploaded file produced no scorable records.")) +
        "</td></tr>";
      return;
    }

    tbody.innerHTML = rows.map(function (app) {
      var scored = app.decision !== "INELIGIBLE" && app.default_probability !== null;
      var imputed = app.imputed_labels && app.imputed_labels.length
        ? '<span class="table-note">' + app.imputed_labels.length + " auto-filled</span>"
        : "";

      return '<tr class="is-clickable" data-id="' + esc(app.id) + '">' +
        '<td><span class="cell-strong">#' + esc(app.id) + "</span>" + imputed + "</td>" +
        '<td class="num">' + esc(fmt.money(app.loan_amnt)) + "</td>" +
        '<td class="num cell-strong">' + (scored ? esc(fmt.pct(app.default_probability)) : fmt.NOT_SCORED) + "</td>" +
        '<td class="num">' + (scored ? esc(app.credit_score) : fmt.NOT_SCORED) + "</td>" +
        '<td class="num">' + (scored ? esc(fmt.money(app.expected_loss)) : fmt.NOT_SCORED) + "</td>" +
        "<td>" + fmt.status(app.decision) + "</td>" +
        '<td class="cell-action"><button class="row-open" type="button" data-action="open-drawer" data-id="' +
        esc(app.id) + '" aria-label="Open applicant ' + esc(app.id) + '">' + icon("chevron-right", 15) + "</button></td>" +
        "</tr>";
    }).join("");
  }

  function renderPagination() {
    var host = dom.qs("#applicants-pagination");
    if (!host) return;
    var total = state.rows.length;
    if (!total) {
      host.innerHTML = "";
      return;
    }
    var start = (state.page - 1) * state.pageSize + 1;
    var end = Math.min(state.page * state.pageSize, total);
    var pages = Math.ceil(total / state.pageSize) || 1;

    host.innerHTML = "<span>" + start + "–" + end + " of " + fmt.count(total) + " applicants</span>" +
      '<div class="pagination__controls">' +
      '<button class="btn btn--secondary btn--sm" type="button" data-action="page" data-dir="-1"' +
      (state.page <= 1 ? " disabled" : "") + ">Previous</button>" +
      '<span class="pagination__label">Page ' + state.page + " of " + pages + "</span>" +
      '<button class="btn btn--secondary btn--sm" type="button" data-action="page" data-dir="1"' +
      (state.page >= pages ? " disabled" : "") + ">Next</button>" +
      "</div>";
  }

  function renderSummaryStrip(data) {
    var host = dom.qs("#summary-strip");
    if (!host) return;
    var s = data.summary;
    var lossRate = s.total_exposure > 0 ? (s.total_expected_loss / s.total_exposure) * 100 : null;

    var cells = [
      { label: "Approval rate", value: s.approval_rate_pct || fmt.EM_DASH },
      { label: "Average PD", value: s.avg_default_probability_pct || fmt.EM_DASH },
      { label: "Average model score", value: s.eligible_count ? fmt.num(s.avg_credit_score, 1) : fmt.NOT_SCORED },
      { label: "Requested exposure", value: fmt.money(s.total_exposure, fmt.EM_DASH) },
      { label: "Expected loss", value: fmt.money(s.total_expected_loss, fmt.EM_DASH) },
      { label: "Expected-loss rate", value: lossRate === null ? fmt.EM_DASH : fmt.pctRaw(lossRate, 2) }
    ];

    host.innerHTML = cells.map(function (cell) {
      return "<div><dt>" + esc(cell.label) + "</dt><dd>" + esc(cell.value) + "</dd></div>";
    }).join("");
  }

  function renderIssues(data) {
    var host = dom.qs("#tab-issues");
    if (!host) return;

    var invalid = data.could_not_score || [];
    var ineligible = (data.applicants || []).filter(function (a) {
      return a.decision === "INELIGIBLE";
    });

    var invalidHtml = '<div class="block">' +
      '<div class="block__head"><h2 class="block__title">Invalid rows</h2>' +
      '<span class="block__note">' + plural(invalid.length, "record") + "</span></div>";

    if (!invalid.length) {
      invalidHtml += stateHtml("No invalid rows", "Every record in this upload passed data validation.");
    } else {
      invalidHtml += '<div class="table-scroll"><table class="data-table"><thead><tr>' +
        "<th>CSV row</th><th>Validation issue</th><th class=\"num\">Loan amount</th>" +
        "<th class=\"num\">Annual income</th><th class=\"num\">DTI</th><th class=\"num\">FICO</th>" +
        "<th>Purpose</th></tr></thead><tbody>" +
        invalid.map(function (row) {
          return "<tr>" +
            '<td><span class="cell-strong">Row ' + esc(row.row_number) + "</span></td>" +
            '<td class="cell-wrap"><span class="field__error">' +
            esc(row.reason_text) + "</span></td>" +
            '<td class="num">' + maybeMoney(row.loan_amnt) + "</td>" +
            '<td class="num">' + maybeMoney(row.annual_inc) + "</td>" +
            '<td class="num">' + maybePct(row.dti) + "</td>" +
            '<td class="num">' + maybeText(row.fico) + "</td>" +
            "<td>" + esc(fmt.humanPurpose(row.purpose)) + "</td>" +
            "</tr>";
        }).join("") + "</tbody></table></div>";
    }
    invalidHtml += "</div>";

    var ineligibleHtml = '<div class="block">' +
      '<div class="block__head"><h2 class="block__title">Ineligible applicants</h2>' +
      '<span class="block__note">' + plural(ineligible.length, "record") + ", not scored</span></div>";

    if (!ineligible.length) {
      ineligibleHtml += stateHtml("No ineligible applicants", "Every record sat inside the 36-month scoring cohort.");
    } else {
      ineligibleHtml += '<div class="table-scroll"><table class="data-table"><thead><tr>' +
        "<th>Applicant</th><th>Term</th><th class=\"num\">Loan amount</th><th>Purpose</th>" +
        "<th>Decision</th><th>Policy warning</th></tr></thead><tbody>" +
        ineligible.map(function (app) {
          return "<tr>" +
            '<td><span class="cell-strong">#' + esc(app.id) + "</span></td>" +
            "<td>" + esc(fmt.termLabel(app.term)) + "</td>" +
            '<td class="num">' + esc(fmt.money(app.loan_amnt)) + "</td>" +
            "<td>" + esc(fmt.humanPurpose(app.purpose)) + "</td>" +
            "<td>" + fmt.status(app.decision) + "</td>" +
            '<td class="cell-wrap cell-wrap--muted">' +
            esc(app.warning || fmt.EM_DASH) + "</td>" +
            "</tr>";
        }).join("") + "</tbody></table></div>";
    }
    ineligibleHtml += "</div>";

    host.innerHTML = invalidHtml + ineligibleHtml;
  }

  /* ----------------------------------------------------------------------
   * Filtering and sorting
   * -------------------------------------------------------------------- */
  function compare(a, b) {
    var field = state.sort.field;
    var dir = state.sort.dir === "asc" ? 1 : -1;
    var va = a[field];
    var vb = b[field];

    var aNil = va === null || va === undefined;
    var bNil = vb === null || vb === undefined;
    if (aNil && bNil) return 0;
    if (aNil) return 1;
    if (bNil) return -1;

    if (typeof va === "string" || typeof vb === "string") {
      return String(va).localeCompare(String(vb)) * dir;
    }
    return (va - vb) * dir;
  }

  function applyFilters() {
    var data = state.data;
    if (!data || !data.applicants) return;

    var query = state.filters.q.trim().toLowerCase().replace(/^#/, "");
    var list = data.applicants.filter(function (app) {
      if (state.filters.decision !== "ALL" && app.decision !== state.filters.decision) return false;
      if (state.filters.band !== "ALL" && app.risk_band !== state.filters.band) return false;
      if (!query) return true;
      return String(app.id).indexOf(query) !== -1 ||
        String(app.purpose || "").toLowerCase().indexOf(query) !== -1;
    });

    list.sort(compare);
    state.rows = list;
    renderApplicantTable();
    renderPagination();
  }

  /* ----------------------------------------------------------------------
   * Drawer
   * -------------------------------------------------------------------- */
  function renderDrawer(app) {
    var host = dom.qs("#drawer");
    if (!host || !app) return;

    var data = state.data || {};
    var scored = app.decision !== "INELIGIBLE" && app.default_probability !== null;
    var raw = app.raw_features || {};
    var tabs = [
      { id: "overview", label: "Overview" },
      { id: "drivers", label: "Drivers" },
      { id: "attributes", label: "Attributes" }
    ];

    host.innerHTML =
      '<div class="drawer__scrim" data-action="close-drawer"></div>' +
      '<div class="drawer__panel" role="dialog" aria-modal="true" aria-labelledby="drawer-title">' +
      '<div class="drawer__head">' +
      '<div class="drawer__identity">' +
      '<div class="drawer__title" id="drawer-title">Applicant #' + esc(app.id) + " " + fmt.status(app.decision) + "</div>" +
      '<div class="drawer__meta">' + esc(fmt.termLabel(app.term)) + " · " + esc(fmt.humanPurpose(app.purpose)) +
      " · " + esc(fmt.money(app.loan_amnt)) + "</div>" +
      "</div>" +
      '<button class="icon-btn" type="button" data-action="close-drawer" aria-label="Close applicant details">' +
      icon("x", 16) + "</button>" +
      "</div>" +
      '<div class="drawer__tabs" role="tablist" aria-label="Applicant detail views">' +
      tabs.map(function (tab) {
        var active = state.drawer.tab === tab.id;
        return '<button class="tabs__item' + (active ? " is-active" : "") + '" type="button" role="tab" ' +
          'aria-selected="' + (active ? "true" : "false") + '" data-action="drawer-tab" data-tab="' + tab.id + '">' +
          esc(tab.label) + "</button>";
      }).join("") +
      "</div>" +
      '<div class="drawer__body" id="drawer-body">' + drawerBody(app, data, scored, raw) + "</div>" +
      "</div>";

    host.hidden = false;
    document.documentElement.classList.add("is-locked");
  }

  function drawerBody(app, data, scored, raw) {
    if (state.drawer.tab === "drivers") {
      return '<div class="block"><div class="block__head"><h3 class="block__title">Feature contributions</h3>' +
        '<span class="block__note">Log-odds, top 4</span></div>' + driverTable(app.reason_codes) + "</div>";
    }

    if (state.drawer.tab === "attributes") {
      return '<div class="block"><div class="block__head"><h3 class="block__title">Submitted attributes</h3>' +
        '<span class="block__note">As uploaded</span></div>' + attributesGrid(raw) + "</div>";
    }

    var html = '<div class="kv-grid">' +
      kv("Model score", scored ? esc(app.credit_score) + " / 850" : fmt.NOT_SCORED) +
      kv("Probability of default", scored ? esc(fmt.pct(app.default_probability)) : fmt.NOT_SCORED) +
      kv("Risk band", fmt.bandDot(app.risk_band, app.decision)) +
      kv("Expected loss", scored ? esc(fmt.money(app.expected_loss)) : fmt.NOT_SCORED) +
      kv("Loan amount", esc(fmt.money(app.loan_amnt))) +
      kv("Bureau FICO", esc(fmt.num(app.fico, 0))) +
      kv("DTI ratio", esc(fmt.pctRaw(app.dti, 1))) +
      kv("Annual income", esc(fmt.money(app.annual_inc))) +
      kv("Purpose", esc(fmt.humanPurpose(app.purpose))) +
      "</div>";

    if (scored) {
      var pos = dom.clamp((Number(app.credit_score) - 300) / 550, 0, 1) * 100;
      html += '<div class="block"><div class="block__head"><h3 class="block__title">Score position</h3>' +
        '<span class="block__note">' + esc(app.risk_band) + "</span></div>" +
        '<div class="score-track"><span class="score-track__marker" style="left:' + pos.toFixed(2) + '%"></span></div>' +
        '<div class="score-track__ends"><span>300</span><span>850</span></div></div>';
    }

    if (app.rationale) {
      html += '<div class="block"><div class="block__head"><h3 class="block__title">Assessment note</h3></div>' +
        '<p class="text-block">' + esc(app.rationale) + "</p></div>";
    }

    if (app.warning) {
      html += '<div class="block"><div class="notice notice--warn">' + noticeHtml("warn", app.warning) + "</div></div>";
    }

    if (app.imputed_labels && app.imputed_labels.length) {
      html += '<div class="block"><div class="block__head"><h3 class="block__title">Auto-filled inputs</h3></div>' +
        '<p class="text-block">' + esc(app.imputed_labels.join(", ")) + "</p></div>";
    }

    if (data.benchmarks) {
      html += '<div class="block"><div class="block__head"><h3 class="block__title">Against portfolio averages</h3></div>' +
        comparisonTable(app, data.benchmarks) + "</div>";
    }

    return html;
  }

  function kv(label, valueHtml) {
    return '<div class="kv"><div class="kv__label">' + esc(label) + "</div>" +
      '<div class="kv__value">' + valueHtml + "</div></div>";
  }

  function destroyDrawer() {
    var host = dom.qs("#drawer");
    if (host) {
      host.hidden = true;
      host.innerHTML = "";
    }
    document.documentElement.classList.remove("is-locked");
  }

  CRC.views = {
    renderAssessment: renderAssessment,
    renderResultsMeta: renderResultsMeta,
    renderToolbarCounts: renderToolbarCounts,
    renderApplicantTable: renderApplicantTable,
    renderPagination: renderPagination,
    renderSummaryStrip: renderSummaryStrip,
    renderIssues: renderIssues,
    applyFilters: applyFilters,
    renderDrawer: renderDrawer,
    destroyDrawer: destroyDrawer,
    stateHtml: stateHtml,
    noticeHtml: noticeHtml
  };
})(window.CRC);
