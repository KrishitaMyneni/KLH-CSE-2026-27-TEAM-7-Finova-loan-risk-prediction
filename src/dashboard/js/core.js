/*
 * core.js — shared primitives for the underwriting console.
 * No dependencies. Everything hangs off window.CRC.
 */
window.CRC = window.CRC || {};
(function (CRC) {
  "use strict";

  /* ----------------------------------------------------------------------
   * Inline SVG icons (lucide path data). No icon font, no emoji, no CDN:
   * the dashboard has to render on an offline host.
   * -------------------------------------------------------------------- */
  var ICONS = {
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    search: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    x: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
    moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "chevron-right": '<polyline points="9 18 15 12 9 6"/>',
    "chevron-down": '<polyline points="6 9 12 15 18 9"/>',
    "chevs-up-down": '<path d="m7 15 5 5 5-5"/><path d="m7 9 5-5 5 5"/>',
    "arrow-up": '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>',
    "arrow-down": '<line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>',
    alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
  };

  function icon(name, size) {
    var body = ICONS[name];
    if (!body) return "";
    var px = size || 16;
    return '<svg class="icon" width="' + px + '" height="' + px + '" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" ' +
      'aria-hidden="true" focusable="false">' + body + '</svg>';
  }

  /** Replace every <span data-icon="name"> in root with its SVG. */
  function hydrateIcons(root) {
    var scope = root || document;
    var nodes = scope.querySelectorAll("[data-icon]");
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      var name = el.getAttribute("data-icon");
      el.innerHTML = icon(name, Number(el.getAttribute("data-icon-size")) || 16);
      el.removeAttribute("data-icon");
    }
  }

  /* ----------------------------------------------------------------------
   * Escaping. reason_text, raw feature values and server error strings are
   * echoed back into markup, so everything passes through here first.
   * -------------------------------------------------------------------- */
  var ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

  function esc(value) {
    if (value === null || value === undefined) return "";
    return String(value).replace(/[&<>"']/g, function (ch) {
      return ESCAPES[ch];
    });
  }

  /* ----------------------------------------------------------------------
   * Formatters. Every number the UI prints goes through one of these so
   * "not scored" never renders as 0.
   * -------------------------------------------------------------------- */
  var NOT_SCORED = "Not scored";
  var EM_DASH = "—";

  function isNil(v) {
    return v === null || v === undefined || v === "";
  }

  function money(value, fallback) {
    if (isNil(value) || isNaN(Number(value))) return fallback === undefined ? NOT_SCORED : fallback;
    return "$" + Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 });
  }

  function moneyCompact(value) {
    var n = Number(value) || 0;
    if (Math.abs(n) >= 1000000) return "$" + (n / 1000000).toFixed(n % 1000000 === 0 ? 0 : 1) + "M";
    if (Math.abs(n) >= 1000) return "$" + Math.round(n / 1000) + "k";
    return "$" + Math.round(n);
  }

  /** Fraction (0.143) -> "14.30%". */
  function pct(frac, digits) {
    if (isNil(frac) || isNaN(Number(frac))) return NOT_SCORED;
    var dp = digits === undefined ? 2 : digits;
    return (Number(frac) * 100).toFixed(dp) + "%";
  }

  /** Ratio already expressed in percent points (16.1) -> "16.1%". */
  function pctRaw(value, digits) {
    // Uploaded values may arrive as "50%" or "50.0"
    var n = isNil(value) ? NaN : Number(String(value).replace("%", "").trim());
    if (isNaN(n)) return EM_DASH;
    var dp = digits === undefined ? 1 : digits;
    return n.toFixed(dp) + "%";
  }

  function count(value) {
    if (isNil(value) || isNaN(Number(value))) return "0";
    return Number(value).toLocaleString("en-US");
  }

  function num(value, digits) {
    if (isNil(value) || isNaN(Number(value))) return EM_DASH;
    var dp = digits === undefined ? 0 : digits;
    return Number(value).toFixed(dp);
  }

  var KEY_LABELS = {
    loan_amnt: "Loan amount",
    term: "Term",
    emp_length: "Employment length",
    home_ownership: "Home ownership",
    annual_inc: "Annual income",
    verification_status: "Income verification",
    purpose: "Loan purpose",
    dti: "DTI ratio",
    delinq_2yrs: "Delinquencies (2 yrs)",
    earliest_cr_line: "Earliest credit line",
    issue_d: "Issue date",
    fico_range_low: "Bureau FICO (low)",
    fico_range_high: "Bureau FICO (high)",
    inq_last_6mths: "Inquiries (6 mths)",
    open_acc: "Open accounts",
    pub_rec: "Public records",
    revol_bal: "Revolving balance",
    revol_util: "Revolving utilization",
    total_acc: "Total accounts",
    initial_list_status: "Listing status",
    application_type: "Application type",
    mort_acc: "Mortgage accounts",
    pub_rec_bankruptcies: "Bankruptcies",
    tot_cur_bal: "Total current balance",
    tot_hi_cred_lim: "Total credit limit",
    bc_util: "Bankcard utilization",
    percent_bc_gt_75: "Bankcards over 75%",
    acc_open_past_24mths: "Accounts opened (24 mths)",
    int_rate: "Interest rate",
    installment: "Installment"
  };

  function humanKey(key) {
    if (KEY_LABELS[key]) return KEY_LABELS[key];
    var text = String(key || "").replace(/_/g, " ");
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  function humanPurpose(value) {
    if (!value) return EM_DASH;
    var text = String(value).replace(/_/g, " ");
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  function termLabel(term) {
    if (!term) return EM_DASH;
    return String(term);
  }

  var DECISION_LABELS = {
    APPROVE: "Approve",
    REVIEW: "Review",
    DECLINE: "Decline",
    INELIGIBLE: "Ineligible"
  };

  function decisionLabel(decision) {
    return DECISION_LABELS[decision] || String(decision || EM_DASH);
  }

  function decisionClass(decision) {
    if (decision === "APPROVE") return "status--approve";
    if (decision === "REVIEW") return "status--review";
    if (decision === "DECLINE") return "status--decline";
    return "status--neutral";
  }

  function bandClass(band) {
    if (band === "Low Risk") return "band-dot--low";
    if (band === "Medium Risk") return "band-dot--med";
    if (band === "High Risk") return "band-dot--high";
    return "band-dot--none";
  }

  function bandLabel(band, decision) {
    if (decision === "INELIGIBLE" || !band || band === "Not scored") return "Not scored";
    return band;
  }

  /** Shared status markup: a 6px dot plus a label. Replaces pill badges. */
  function status(decision, large) {
    return '<span class="status ' + decisionClass(decision) + (large ? " status--lg" : "") + '">' +
      '<span class="status__dot" aria-hidden="true"></span>' +
      esc(decisionLabel(decision)) + "</span>";
  }

  function bandDot(band, decision) {
    return '<span class="band-dot ' + bandClass(band) + '">' +
      '<span class="band-dot__mark" aria-hidden="true"></span>' +
      esc(bandLabel(band, decision)) + "</span>";
  }

  /* ----------------------------------------------------------------------
   * DOM helpers
   * -------------------------------------------------------------------- */
  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }

  function qsa(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  function on(el, type, handler, options) {
    if (el) el.addEventListener(type, handler, options || false);
  }

  function debounce(fn, wait) {
    var timer = null;
    return function () {
      var args = arguments;
      var ctx = this;
      window.clearTimeout(timer);
      timer = window.setTimeout(function () {
        fn.apply(ctx, args);
      }, wait);
    };
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  /** Read a design token so charts always follow the active theme. */
  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ----------------------------------------------------------------------
   * Application state. One object, no framework.
   * -------------------------------------------------------------------- */
  var state = {
    mode: "single",
    view: "application",
    tab: "applicants",
    data: null,
    rows: [],
    filters: { q: "", decision: "ALL", band: "ALL" },
    sort: { field: "default_probability", dir: "desc" },
    page: 1,
    pageSize: 25,
    assessment: null,
    drawer: { open: false, tab: "overview", id: null },
    pending: false,
    lastSubmitted: null
  };

  CRC.icon = icon;
  CRC.hydrateIcons = hydrateIcons;
  CRC.esc = esc;
  CRC.fmt = {
    NOT_SCORED: NOT_SCORED,
    EM_DASH: EM_DASH,
    isNil: isNil,
    money: money,
    moneyCompact: moneyCompact,
    pct: pct,
    pctRaw: pctRaw,
    count: count,
    num: num,
    humanKey: humanKey,
    humanPurpose: humanPurpose,
    termLabel: termLabel,
    decisionLabel: decisionLabel,
    bandLabel: bandLabel,
    status: status,
    bandDot: bandDot
  };
  CRC.dom = { qs: qs, qsa: qsa, on: on, debounce: debounce, clamp: clamp, token: token };
  CRC.state = state;
})(window.CRC);
