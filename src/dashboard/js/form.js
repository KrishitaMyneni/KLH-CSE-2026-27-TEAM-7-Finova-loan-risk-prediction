/*
 * form.js — single-applicant form.
 *
 * The backend has one scoring entry point (POST /api/score, multipart CSV), so
 * this module validates the form and composes a one-row CSV in the exact column
 * order of data/template_applicants.csv.
 *
 * percent_bc_gt_75 is not in app.REQUIRED_COLUMNS, but the champion's
 * numeric_features include it, so it must be sent or app.py raises on
 * df_feat[numeric_features]. It is therefore part of the schema below.
 */
window.CRC = window.CRC || {};
(function (CRC) {
  "use strict";

  var fmt = CRC.fmt;
  var dom = CRC.dom;

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  var EMP_OPTIONS = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10+ years"
  ];

  var PURPOSE_OPTIONS = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "medical", "small_business", "car", "moving", "vacation", "house", "other"
  ];

  var SECTIONS = [
    { id: "loan", title: "Loan request", advanced: false },
    { id: "borrower", title: "Borrower", advanced: false },
    { id: "bureau", title: "Credit bureau", advanced: false },
    { id: "accounts", title: "Accounts and balances", advanced: false },
    { id: "platform", title: "Platform fields", advanced: true }
  ];

  /*
   * Field schema. `csv` is the uploaded column name; `default` mirrors the
   * example row shipped in data/template_applicants.csv.
   */
  var FIELDS = [
    { key: "loan_amnt", csv: "loan_amnt", section: "loan", label: "Loan amount", type: "money",
      min: 1000, max: 40000, step: 500, unit: "USD", hint: "1,000 – 40,000", default: 15000 },

    { key: "term", csv: "term", section: "loan", label: "Term", type: "select",
      options: [{ value: "36 months", label: "36 months" }, { value: "60 months", label: "60 months" }],
      hint: "Only 36-month loans are scored", default: "36 months" },

    { key: "purpose", csv: "purpose", section: "loan", label: "Purpose", type: "select",
      options: PURPOSE_OPTIONS.map(function (p) {
        return { value: p, label: fmt.humanPurpose(p) };
      }), default: "debt_consolidation" },

    { key: "annual_inc", csv: "annual_inc", section: "borrower", label: "Annual income", type: "money",
      min: 0, max: 10000000, step: 1000, unit: "USD", hint: "Gross income before tax", default: 75000 },

    { key: "emp_length", csv: "emp_length", section: "borrower", label: "Employment length", type: "select",
      options: EMP_OPTIONS.map(function (e) { return { value: e, label: e }; }), default: "5 years" },

    { key: "home_ownership", csv: "home_ownership", section: "borrower", label: "Home ownership", type: "select",
      options: [
        { value: "MORTGAGE", label: "Mortgage" },
        { value: "OWN", label: "Own outright" },
        { value: "RENT", label: "Rent" }
      ], default: "MORTGAGE" },

    { key: "verification_status", csv: "verification_status", section: "borrower", label: "Income verification", type: "select",
      options: [
        { value: "Not Verified", label: "Not verified" },
        { value: "Source Verified", label: "Source verified" },
        { value: "Verified", label: "Verified" }
      ], default: "Source Verified" },

    { key: "fico_range_low", csv: "fico_range_low", section: "bureau", label: "Bureau FICO", type: "number",
      min: 300, max: 850, step: 1, hint: "300 – 850. The upper range is derived.", default: 720 },

    { key: "earliest_cr_line", csv: "earliest_cr_line", section: "bureau", label: "Earliest credit line", type: "month-year",
      hint: "Month and year of the oldest account", defaultMonth: "Jan", defaultYear: 2010 },

    { key: "dti", csv: "dti", section: "bureau", label: "DTI ratio", type: "percent",
      min: 0, max: 100, step: 0.1, hint: "Monthly debt payment ÷ monthly income", default: 14.5 },

    { key: "delinq_2yrs", csv: "delinq_2yrs", section: "bureau", label: "Delinquencies (2 yrs)", type: "number",
      min: 0, max: 100, step: 1, default: 0 },

    { key: "inq_last_6mths", csv: "inq_last_6mths", section: "bureau", label: "Inquiries (6 mths)", type: "number",
      min: 0, max: 100, step: 1, default: 0 },

    { key: "pub_rec", csv: "pub_rec", section: "bureau", label: "Public records", type: "number",
      min: 0, max: 100, step: 1, default: 0 },

    { key: "pub_rec_bankruptcies", csv: "pub_rec_bankruptcies", section: "bureau", label: "Bankruptcies", type: "number",
      min: 0, max: 20, step: 1, default: 0 },

    { key: "revol_bal", csv: "revol_bal", section: "accounts", label: "Revolving balance", type: "money",
      min: 0, max: 10000000, step: 100, unit: "USD", default: 12000 },

    { key: "revol_util", csv: "revol_util", section: "accounts", label: "Revolving utilization", type: "percent",
      min: 0, max: 150, step: 0.1, default: 35 },

    { key: "bc_util", csv: "bc_util", section: "accounts", label: "Bankcard utilization", type: "percent",
      min: 0, max: 150, step: 0.1, default: 32 },

    { key: "percent_bc_gt_75", csv: "percent_bc_gt_75", section: "accounts", label: "Bankcards over 75%", type: "percent",
      min: 0, max: 100, step: 0.1, hint: "Share of bankcards above 75% of limit", default: 0 },

    { key: "tot_cur_bal", csv: "tot_cur_bal", section: "accounts", label: "Total current balance", type: "money",
      min: 0, max: 100000000, step: 1000, unit: "USD", default: 185000 },

    { key: "tot_hi_cred_lim", csv: "tot_hi_cred_lim", section: "accounts", label: "Total credit limit", type: "money",
      min: 0, max: 100000000, step: 1000, unit: "USD", default: 240000 },

    { key: "open_acc", csv: "open_acc", section: "accounts", label: "Open accounts", type: "number",
      min: 0, max: 100, step: 1, default: 10 },

    { key: "total_acc", csv: "total_acc", section: "accounts", label: "Total accounts", type: "number",
      min: 0, max: 200, step: 1, default: 20 },

    { key: "mort_acc", csv: "mort_acc", section: "accounts", label: "Mortgage accounts", type: "number",
      min: 0, max: 100, step: 1, default: 1 },

    { key: "acc_open_past_24mths", csv: "acc_open_past_24mths", section: "accounts", label: "Accounts opened (24 mths)", type: "number",
      min: 0, max: 100, step: 1, default: 2 },

    { key: "initial_list_status", csv: "initial_list_status", section: "platform", label: "Listing status", type: "select",
      options: [{ value: "w", label: "Whole" }, { value: "f", label: "Fractional" }], default: "w" },

    { key: "application_type", csv: "application_type", section: "platform", label: "Application type", type: "select",
      options: [{ value: "Individual", label: "Individual" }, { value: "Joint App", label: "Joint" }],
      default: "Individual" }
  ];

  /* Column order matches data/template_applicants.csv, minus issue_d (the API
   * defaults it) and including percent_bc_gt_75. */
  var CSV_COLUMNS = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status", "application_type",
    "mort_acc", "pub_rec_bankruptcies", "tot_cur_bal", "tot_hi_cred_lim", "bc_util",
    "percent_bc_gt_75", "acc_open_past_24mths"
  ];

  var fieldByKey = {};
  FIELDS.forEach(function (f) { fieldByKey[f.key] = f; });

  /* ----------------------------------------------------------------------
   * Rendering
   * -------------------------------------------------------------------- */
  function fieldHtml(field) {
    var describedBy = "hint-" + field.key;
    var label = '<label class="field__label" for="f-' + field.key + '">' + CRC.esc(field.label) +
      (field.unit ? ' <span class="field__unit">' + CRC.esc(field.unit) + "</span>" : "") + "</label>";
    var control;

    if (field.type === "select") {
      control = '<select class="select field__control" id="f-' + field.key + '" name="' + field.key + '">' +
        field.options.map(function (opt) {
          return '<option value="' + CRC.esc(opt.value) + '"' +
            (opt.value === field.default ? " selected" : "") + ">" + CRC.esc(opt.label) + "</option>";
        }).join("") + "</select>";
    } else if (field.type === "month-year") {
      var nowYear = new Date().getFullYear();
      var years = [];
      for (var y = nowYear; y >= 1950; y--) years.push(y);
      var currentYear = String(field.defaultYear);
      return '<div class="field" data-field="' + field.key + '">' +
        '<span class="field__label" id="lbl-' + field.key + '">' + CRC.esc(field.label) + "</span>" +
        '<div class="field__pair">' +
        '<select class="select field__control" id="f-earliest_month" aria-labelledby="lbl-' + field.key + '">' +
        MONTHS.map(function (m) {
          return '<option value="' + m + '"' + (m === field.defaultMonth ? " selected" : "") + ">" + m + "</option>";
        }).join("") + "</select>" +
        '<select class="select field__control" id="f-earliest_year" aria-labelledby="lbl-' + field.key + '">' +
        years.map(function (yr) {
          return '<option value="' + yr + '"' + (String(yr) === currentYear ? " selected" : "") + ">" + yr + "</option>";
        }).join("") + "</select>" +
        "</div>" +
        '<span class="field__hint" id="' + describedBy + '">' + CRC.esc(field.hint || "") + "</span>" +
        "</div>";
    } else {
      control = '<input class="input field__control" id="f-' + field.key + '" name="' + field.key +
        '" type="number" inputmode="decimal"' +
        ' min="' + field.min + '" max="' + field.max + '" step="' + field.step + '"' +
        ' value="' + field.default + '" aria-describedby="' + describedBy + '">';
    }

    return '<div class="field" data-field="' + field.key + '">' +
      label + control +
      (field.hint ? '<span class="field__hint" id="' + describedBy + '">' + CRC.esc(field.hint) + "</span>" : "") +
      "</div>";
  }

  function renderForm() {
    var host = dom.qs("#form-sections");
    if (!host) return;

    var html = SECTIONS.filter(function (s) { return !s.advanced; }).map(function (section) {
      var fields = FIELDS.filter(function (f) { return f.section === section.id; });
      return '<fieldset class="form-section">' +
        '<legend class="form-section__legend">' + CRC.esc(section.title) + "</legend>" +
        '<div class="form-grid">' + fields.map(fieldHtml).join("") + "</div>" +
        "</fieldset>";
    }).join("");

    var advanced = SECTIONS.filter(function (s) { return s.advanced; })[0];
    var advancedFields = FIELDS.filter(function (f) { return f.section === advanced.id; });
    html += '<details class="advanced" id="advanced-fields">' +
      '<summary class="advanced__summary">' +
      CRC.icon("chevron-right", 14) +
      CRC.esc(advanced.title) +
      "</summary>" +
      '<div class="advanced__body"><div class="form-grid">' +
      advancedFields.map(fieldHtml).join("") + "</div></div>" +
      "</details>";

    host.innerHTML = html;
  }

  /* ----------------------------------------------------------------------
   * Reading and writing values
   * -------------------------------------------------------------------- */
  function readValues() {
    var values = {};
    FIELDS.forEach(function (f) {
      if (f.type === "month-year") {
        var month = dom.qs("#f-earliest_month");
        var year = dom.qs("#f-earliest_year");
        values[f.key] = (month ? month.value : f.defaultMonth) + "-" + (year ? year.value : f.defaultYear);
        return;
      }
      var el = dom.qs("#f-" + f.key);
      values[f.key] = el ? el.value : "";
    });
    return values;
  }

  function applyValues(values) {
    if (!values) return;
    FIELDS.forEach(function (f) {
      if (f.type === "month-year") {
        var parts = String(values[f.key] || "").split("-");
        var monthEl = dom.qs("#f-earliest_month");
        var yearEl = dom.qs("#f-earliest_year");
        if (monthEl && parts[0]) monthEl.value = parts[0];
        if (yearEl && parts[1]) yearEl.value = parts[1];
        return;
      }
      var el = dom.qs("#f-" + f.key);
      if (el && values[f.key] !== undefined) el.value = values[f.key];
    });
  }

  function resetToExample() {
    var values = {};
    FIELDS.forEach(function (f) {
      values[f.key] = f.type === "month-year"
        ? f.defaultMonth + "-" + f.defaultYear
        : f.default;
    });
    applyValues(values);
    clearErrors();
  }

  /* ----------------------------------------------------------------------
   * Validation. Backend rules are authoritative; these mirror the ones
   * app.py enforces (loan_amnt > 0, annual_inc >= 0, dti 0-100, fico 300-850)
   * so the user gets an inline message instead of a 422 round trip.
   * -------------------------------------------------------------------- */
  function bound(field, n) {
    var text = Number(n).toLocaleString("en-US");
    if (field.type === "money") return "$" + text;
    if (field.type === "percent") return text + "%";
    return text;
  }

  function validate(values) {
    var errors = {};
    var order = [];

    function fail(key, message) {
      errors[key] = message;
      order.push(key);
    }

    FIELDS.forEach(function (field) {
      if (field.type === "select" || field.type === "month-year") return;
      var raw = values[field.key];
      var value = Number(raw);
      var label = field.label;

      if (raw === "" || raw === null || raw === undefined || isNaN(value)) {
        // Lower-case the first word unless it is an acronym ("DTI ratio", "FICO")
        var phrase = /^[A-Z]{2}/.test(label) ? label : label.charAt(0).toLowerCase() + label.slice(1);
        fail(field.key, "Enter " + (/^[aeiou]/i.test(phrase) ? "an " : "a ") + phrase + ".");
        return;
      }
      if (value < field.min || value > field.max) {
        fail(field.key, label + " must be between " + bound(field, field.min) + " and " +
          bound(field, field.max) + ".");
        return;
      }
      if (field.type === "number" && value % 1 !== 0) {
        fail(field.key, label + " must be a whole number.");
      }
    });

    // Cross-field rules the backend would otherwise accept silently
    if (!errors.open_acc && !errors.total_acc && Number(values.open_acc) > Number(values.total_acc)) {
      fail("open_acc", "Open accounts can't be more than total accounts.");
    }
    var now = new Date();
    var parts = String(values.earliest_cr_line || "").split("-");
    var lineMonth = MONTHS.indexOf(parts[0]);
    var lineYear = Number(parts[1]);
    if (lineYear > now.getFullYear() || (lineYear === now.getFullYear() && lineMonth > now.getMonth())) {
      fail("earliest_cr_line", "The earliest credit line can't be in the future.");
    }

    return { valid: order.length === 0, errors: errors, order: order };
  }

  function clearErrors() {
    dom.qsa(".field--invalid").forEach(function (el) {
      el.classList.remove("field--invalid");
    });
    dom.qsa(".field__error").forEach(function (el) {
      el.remove();
    });
    dom.qsa("[aria-invalid]").forEach(function (el) {
      el.removeAttribute("aria-invalid");
      var original = el.getAttribute("data-describedby");
      if (original !== null) {
        if (original) el.setAttribute("aria-describedby", original);
        else el.removeAttribute("aria-describedby");
        el.removeAttribute("data-describedby");
      }
    });
    var summary = dom.qs("#form-summary");
    if (summary) {
      summary.hidden = true;
      summary.innerHTML = "";
    }
  }

  function applyErrors(result) {
    clearErrors();
    if (result.valid) return;

    result.order.forEach(function (key) {
      var field = fieldByKey[key];
      var wrapper = dom.qs('.field[data-field="' + key + '"]');
      if (!wrapper || !field) return;
      wrapper.classList.add("field--invalid");

      var target = dom.qs("#f-" + key) || dom.qs("#f-earliest_month");
      if (target) target.setAttribute("aria-invalid", "true");

      var message = document.createElement("span");
      message.className = "field__error";
      message.id = "err-" + key;
      message.textContent = result.errors[key];
      wrapper.appendChild(message);

      if (target) {
        var hint = target.getAttribute("aria-describedby") || "";
        target.setAttribute("data-describedby", hint);
        target.setAttribute("aria-describedby", (hint ? hint + " " : "") + "err-" + key);
      }
    });

    var summary = dom.qs("#form-summary");
    if (summary) {
      var links = result.order.slice(0, 4).map(function (key) {
        return '<a href="#f-' + key + '" data-focus-field="' + key + '">' + CRC.esc(fieldByKey[key].label) + "</a>";
      }).join(", ");
      var more = result.order.length > 4 ? " and " + (result.order.length - 4) + " more" : "";
      summary.innerHTML = '<span class="notice__icon">' + CRC.icon("alert", 15) + "</span>" +
        "<span>" + result.order.length + " field" + (result.order.length === 1 ? "" : "s") +
        " need attention: " + links + more + "</span>";
      summary.hidden = false;
    }

    var first = dom.qs("[aria-invalid='true']");
    if (first) first.focus();
  }

  /* ----------------------------------------------------------------------
   * CSV composition
   * -------------------------------------------------------------------- */
  function csvEscape(value) {
    var text = value === null || value === undefined ? "" : String(value);
    if (/[",\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
    return text;
  }

  function toRow(values) {
    var row = {};
    var ficoLow = Math.round(Number(values.fico_range_low) || 0);
    var ficoHigh = Math.min(850, ficoLow + 4);

    FIELDS.forEach(function (field) {
      var value = values[field.key];
      if (field.type === "number" || field.type === "money" || field.type === "percent") {
        value = Number(value);
      }
      row[field.csv] = value;
    });
    row.fico_range_low = ficoLow;
    row.fico_range_high = ficoHigh;
    return row;
  }

  function composeFile(values) {
    var row = toRow(values);
    var header = CSV_COLUMNS.join(",");
    var line = CSV_COLUMNS.map(function (col) {
      return csvEscape(row[col]);
    }).join(",");
    var csv = header + "\n" + line + "\n";
    return new File([csv], "applicant.csv", { type: "text/csv" });
  }

  CRC.form = {
    FIELDS: FIELDS,
    CSV_COLUMNS: CSV_COLUMNS,
    renderForm: renderForm,
    readValues: readValues,
    applyValues: applyValues,
    resetToExample: resetToExample,
    validate: validate,
    applyErrors: applyErrors,
    clearErrors: clearErrors,
    composeFile: composeFile,
    csvEscape: csvEscape
  };
})(window.CRC);
