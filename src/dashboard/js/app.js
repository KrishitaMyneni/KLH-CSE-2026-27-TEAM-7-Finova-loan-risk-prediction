/*
 * app.js — controllers and event wiring.
 *
 * Backend contract (frozen): POST /api/score takes one CSV file and returns
 * summary + applicants + could_not_score. Everything here is presentation
 * state on top of that single call.
 */
(function (CRC) {
  "use strict";

  var dom = CRC.dom;
  var fmt = CRC.fmt;
  var esc = CRC.esc;
  var state = CRC.state;
  var views = CRC.views;
  var form = CRC.form;

  var lastFocused = null;

  /* ----------------------------------------------------------------------
   * API
   * -------------------------------------------------------------------- */
  function readErrorDetail(payload) {
    if (!payload) return null;
    var detail = payload.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map(function (item) {
        return item && item.msg ? item.msg : JSON.stringify(item);
      }).join("\n");
    }
    if (payload.message) return String(payload.message);
    return null;
  }

  async function postScore(file) {
    var body = new FormData();
    body.append("file", file);

    var response;
    try {
      response = await fetch("/api/score", { method: "POST", body: body });
    } catch (err) {
      throw new Error("Network error — the scoring service could not be reached.");
    }

    var payload = null;
    try {
      payload = await response.json();
    } catch (err) {
      payload = null;
    }

    if (!response.ok) {
      var message = readErrorDetail(payload) || ("Scoring failed with status " + response.status + ".");
      throw new Error(message);
    }
    if (!payload || payload.status !== "success") {
      throw new Error("The scoring service returned an unexpected response.");
    }
    return payload;
  }

  async function fetchCsvAsFile(url, filename) {
    var response = await fetch(url);
    if (!response.ok) throw new Error("Could not load " + filename + ".");
    var blob = await response.blob();
    return new File([blob], filename, { type: "text/csv" });
  }

  /* ----------------------------------------------------------------------
   * Mode and view routing
   * -------------------------------------------------------------------- */
  function setTitle(text) {
    var el = dom.qs("#page-title");
    if (el) el.textContent = text;
  }

  function setMode(mode) {
    state.mode = mode;

    dom.qsa("#mode-switch .mode-switch__item").forEach(function (btn) {
      var active = btn.getAttribute("data-mode") === mode;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });

    var application = dom.qs("#view-application");
    var assessment = dom.qs("#view-assessment");
    var batch = dom.qs("#view-batch");
    var meta = dom.qs("#page-meta");
    if (meta) meta.hidden = true;

    if (mode === "single") {
      batch.hidden = true;
      var hasAssessment = Boolean(state.assessment);
      application.hidden = hasAssessment;
      assessment.hidden = !hasAssessment;
      setTitle(hasAssessment ? "Assessment" : "New applicant");
      return;
    }

    application.hidden = true;
    assessment.hidden = true;
    batch.hidden = false;

    var hasBatch = Boolean(state.data);
    dom.qs("#batch-input").hidden = hasBatch;
    dom.qs("#batch-results").hidden = !hasBatch;
    setTitle(hasBatch ? "Batch results" : "Upload batch");
  }

  function setTab(tab) {
    state.tab = tab;

    dom.qsa("#batch-tabs .tabs__item").forEach(function (btn) {
      var active = btn.getAttribute("data-tab") === tab;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });

    ["applicants", "portfolio", "issues"].forEach(function (name) {
      var panel = dom.qs("#tab-" + name);
      if (panel) panel.hidden = name !== tab;
    });

    if (tab === "portfolio") renderPortfolio();
  }

  function tabLabelWithCount(tab, count) {
    var btn = dom.qs('#batch-tabs .tabs__item[data-tab="' + tab + '"]');
    if (!btn) return;
    var base = tab === "issues" ? "Input issues" : "Applicants";
    btn.innerHTML = esc(base) + (count > 0 ? '<span class="tabs__count">' + count + "</span>" : "");
  }

  /* ----------------------------------------------------------------------
   * Single applicant
   * -------------------------------------------------------------------- */
  function setSubmitPending(on) {
    var button = dom.qs("#submit-applicant");
    if (!button) return;
    state.pending = on;
    button.disabled = on;
    button.setAttribute("aria-busy", on ? "true" : "false");
    button.innerHTML = on
      ? '<span class="spinner" aria-hidden="true"></span>Scoring…'
      : "Score applicant";
  }

  function showFormError(message) {
    var el = dom.qs("#form-error");
    if (!el) return;
    el.innerHTML = views.noticeHtml("error", message);
    el.hidden = false;
  }

  function clearFormError() {
    var el = dom.qs("#form-error");
    if (el) {
      el.hidden = true;
      el.innerHTML = "";
    }
  }

  async function submitApplication(event) {
    if (event) event.preventDefault();
    if (state.pending) return;

    var values = form.readValues();
    state.lastSubmitted = values;
    clearFormError();

    var result = form.validate(values);
    if (!result.valid) {
      form.applyErrors(result);
      return;
    }
    form.clearErrors();

    var file = form.composeFile(values);
    setSubmitPending(true);

    try {
      var data = await postScore(file);
      var app = (data.applicants || [])[0];
      if (!app) throw new Error("No applicant was returned by the scoring service.");
      state.assessment = { data: data, app: app };
      views.renderAssessment(data, app);
      setMode("single");
      window.scrollTo({ top: 0, behavior: "auto" });
    } catch (err) {
      showFormError(err.message);
    } finally {
      setSubmitPending(false);
    }
  }

  /* ----------------------------------------------------------------------
   * Batch
   * -------------------------------------------------------------------- */
  function setBatchPending(on, label) {
    var control = dom.qs("#file-control");
    var browse = control ? control.querySelector(".btn") : null;
    if (!control || !browse) return;
    control.classList.toggle("is-busy", on);
    if (on) {
      browse.innerHTML = '<span class="spinner" aria-hidden="true"></span>' + esc(label || "Scoring");
      control.setAttribute("aria-busy", "true");
    } else {
      browse.textContent = "Browse";
      control.removeAttribute("aria-busy");
    }
  }

  function showBatchError(message) {
    var el = dom.qs("#batch-error");
    if (!el) return;
    el.innerHTML = views.noticeHtml("error", message);
    el.hidden = false;
  }

  function clearBatchError() {
    var el = dom.qs("#batch-error");
    if (el) {
      el.hidden = true;
      el.innerHTML = "";
    }
  }

  function resetResultsFilters() {
    state.filters = { q: "", decision: "ALL", band: "ALL" };
    state.page = 1;
    var search = dom.qs("#search-input");
    if (search) search.value = "";
    var band = dom.qs("#band-filter");
    if (band) band.value = "ALL";
  }

  function renderBatch(data) {
    state.data = data;
    resetResultsFilters();

    dom.qs("#batch-input").hidden = true;
    dom.qs("#batch-results").hidden = false;

    views.renderResultsMeta(data);
    views.renderToolbarCounts(data);
    views.renderSummaryStrip(data);
    views.renderIssues(data);
    views.applyFilters();

    var warning = dom.qs("#batch-warning");
    if (warning) {
      if (data.global_warning) {
        warning.innerHTML = views.noticeHtml("warn", data.global_warning);
        warning.hidden = false;
      } else {
        warning.hidden = true;
        warning.innerHTML = "";
      }
    }

    tabLabelWithCount("issues", data.summary.could_not_score_count);
    setTab("applicants");
    setMode("batch");
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  var batchBusy = false;

  async function scoreBatchFile(file) {
    if (!file || batchBusy) return;
    clearBatchError();

    if (!/\.csv$/i.test(file.name)) {
      showBatchError("The file must have a .csv extension.");
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      showBatchError("The file is larger than the 5 MB upload limit.");
      return;
    }

    batchBusy = true;
    setBatchPending(true, "Scoring " + file.name);
    try {
      var data = await postScore(file);
      renderBatch(data);
    } catch (err) {
      showBatchError(err.message);
    } finally {
      batchBusy = false;
      setBatchPending(false);
    }
  }

  async function loadDemo(url, filename) {
    if (batchBusy) return;
    batchBusy = true; // claim the slot before the download, not after
    clearBatchError();
    setBatchPending(true, "Loading " + filename);
    try {
      var file = await fetchCsvAsFile(url, filename);
      batchBusy = false; // hand over to scoreBatchFile in the same tick
      await scoreBatchFile(file);
    } catch (err) {
      batchBusy = false;
      showBatchError(err.message);
      setBatchPending(false);
    }
  }

  /* ----------------------------------------------------------------------
   * Portfolio tab (charts render lazily, only when visible)
   * -------------------------------------------------------------------- */
  function renderPortfolio() {
    var data = state.data;
    var grid = dom.qs("#chart-grid");
    if (!data || !grid) return;

    var hasEligible = data.summary.eligible_count > 0;
    var placeholder = dom.qs("#portfolio-state");
    if (!placeholder) {
      placeholder = document.createElement("div");
      placeholder.id = "portfolio-state";
      grid.parentNode.insertBefore(placeholder, grid.nextSibling);
    }

    if (!hasEligible) {
      CRC.charts.destroyAll();
      placeholder.innerHTML = views.stateHtml(
        "No scored applicants",
        "Every record in this upload was excluded by the 36-month policy, so there is no portfolio to chart."
      );
      placeholder.hidden = false;
      grid.hidden = true;
      return;
    }

    placeholder.hidden = true;
    placeholder.innerHTML = "";
    grid.hidden = false;
    CRC.charts.renderAll(data);
  }

  /* ----------------------------------------------------------------------
   * Drawer
   * -------------------------------------------------------------------- */
  function openDrawer(id) {
    var data = state.data;
    if (!data) return;
    var app = (data.applicants || []).filter(function (row) {
      return String(row.id) === String(id);
    })[0];
    if (!app) return;

    lastFocused = document.activeElement;
    state.drawer.open = true;
    state.drawer.id = id;
    state.drawer.tab = "overview";
    views.renderDrawer(app);

    var closeBtn = dom.qs('#drawer [data-action="close-drawer"].icon-btn');
    if (closeBtn) closeBtn.focus();
  }

  function closeDrawer() {
    if (!state.drawer.open) return;
    state.drawer.open = false;
    state.drawer.id = null;
    views.destroyDrawer();
    if (lastFocused && document.contains(lastFocused)) lastFocused.focus();
  }

  function trapFocus(event) {
    if (event.key !== "Tab" || !state.drawer.open) return;
    var panel = dom.qs(".drawer__panel");
    if (!panel) return;
    var focusable = dom.qsa(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      panel
    ).filter(function (el) {
      return !el.disabled && el.offsetParent !== null;
    });
    if (!focusable.length) return;

    var first = focusable[0];
    var last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  /* ----------------------------------------------------------------------
   * Export
   * -------------------------------------------------------------------- */
  function exportCsv() {
    var data = state.data;
    if (!data || !data.applicants) return;

    var headers = [
      "applicant_id", "loan_amnt", "term", "annual_inc", "bureau_fico", "dti",
      "default_probability", "model_score", "risk_band", "expected_loss", "decision"
    ];

    var lines = [headers.join(",")];
    data.applicants.forEach(function (app) {
      lines.push([
        app.id,
        app.loan_amnt,
        form.csvEscape(app.term),
        app.annual_inc,
        app.fico,
        app.dti,
        app.default_probability === null ? "" : app.default_probability,
        app.credit_score === null ? "" : app.credit_score,
        form.csvEscape(app.risk_band),
        app.expected_loss === null ? "" : app.expected_loss,
        form.csvEscape(app.decision)
      ].join(","));
    });

    var blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "scored_applicants.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /* ----------------------------------------------------------------------
   * Theme
   * -------------------------------------------------------------------- */
  function applyThemeIcon() {
    var button = dom.qs("#theme-toggle");
    if (!button) return;
    var isDark = document.documentElement.getAttribute("data-theme") === "dark";
    button.innerHTML = CRC.icon(isDark ? "sun" : "moon", 16);
    button.setAttribute("aria-label", isDark ? "Switch to light theme" : "Switch to dark theme");
  }

  function toggleTheme() {
    var root = document.documentElement;
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("crc-theme", next);
    } catch (err) { /* storage unavailable */ }
    applyThemeIcon();

    if (state.data && state.tab === "portfolio" && !dom.qs("#chart-grid").hidden) {
      CRC.charts.renderAll(state.data);
    }
  }

  /* ----------------------------------------------------------------------
   * Health
   * -------------------------------------------------------------------- */
  function initHealth() {
    var el = dom.qs("#model-note");
    if (!el) return;
    fetch("/api/health")
      .then(function (res) {
        if (!res.ok) throw new Error("health failed");
        return res.json();
      })
      .then(function (health) {
        var auc = typeof health.test_roc_auc === "number"
          ? health.test_roc_auc.toFixed(3)
          : String(health.test_roc_auc);
        el.textContent = health.model + " · test AUC " + auc;
        el.title = "Champion model, isotonic-calibrated. Test window AUC " + auc + ".";
      })
      .catch(function () {
        el.textContent = "Model details unavailable";
      });
  }

  /* ----------------------------------------------------------------------
   * Events — one delegated click handler covers re-rendered regions.
   * -------------------------------------------------------------------- */
  function bindEvents() {
    CRC.dom.on(document, "click", function (event) {
      var actionEl = event.target.closest("[data-action]");
      if (actionEl) {
        var action = actionEl.getAttribute("data-action");

        if (action === "sort") {
          var field = actionEl.getAttribute("data-field");
          if (state.sort.field === field) {
            state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
          } else {
            state.sort.field = field;
            state.sort.dir = field === "default_probability" || field === "expected_loss" ? "desc" : "asc";
          }
          views.applyFilters();
          return;
        }

        if (action === "filter-decision") {
          state.filters.decision = actionEl.getAttribute("data-decision");
          state.page = 1;
          views.renderToolbarCounts(state.data);
          views.applyFilters();
          return;
        }

        if (action === "page") {
          var dir = Number(actionEl.getAttribute("data-dir"));
          var pages = Math.ceil(state.rows.length / state.pageSize) || 1;
          state.page = dom.clamp(state.page + dir, 1, pages);
          views.renderApplicantTable();
          views.renderPagination();
          return;
        }

        if (action === "clear-filters") {
          resetResultsFilters();
          views.renderToolbarCounts(state.data);
          views.applyFilters();
          return;
        }

        if (action === "open-drawer") {
          openDrawer(actionEl.getAttribute("data-id"));
          return;
        }

        if (action === "close-drawer") {
          closeDrawer();
          return;
        }

        if (action === "drawer-tab") {
          state.drawer.tab = actionEl.getAttribute("data-tab");
          var data = state.data;
          var app = (data.applicants || []).filter(function (row) {
            return String(row.id) === String(state.drawer.id);
          })[0];
          if (app) {
            views.renderDrawer(app);
            // The drawer re-renders, so put focus back on the tab just chosen
            var activeTab = dom.qs('#drawer .tabs__item.is-active');
            if (activeTab) activeTab.focus();
          }
          return;
        }

        if (action === "edit-inputs") {
          state.assessment = null;
          if (state.lastSubmitted) form.applyValues(state.lastSubmitted);
          setMode("single");
          return;
        }

        if (action === "new-assessment") {
          state.assessment = null;
          form.resetToExample();
          form.clearErrors();
          clearFormError();
          setMode("single");
          window.scrollTo({ top: 0, behavior: "auto" });
          return;
        }
      }

      // Row click opens the drawer, except when a control inside the row handled it.
      var row = event.target.closest("tr[data-id]");
      if (row && !event.target.closest("button")) {
        openDrawer(row.getAttribute("data-id"));
      }
    });

    dom.qsa("#mode-switch .mode-switch__item").forEach(function (btn) {
      dom.on(btn, "click", function () {
        setMode(btn.getAttribute("data-mode"));
      });
    });

    dom.qsa("#batch-tabs .tabs__item").forEach(function (btn) {
      dom.on(btn, "click", function () {
        setTab(btn.getAttribute("data-tab"));
      });
    });

    var tablists = [dom.qs("#mode-switch"), dom.qs("#batch-tabs")];
    tablists.forEach(function (list) {
      if (!list) return;
      dom.on(list, "keydown", function (event) {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        var items = dom.qsa('[role="tab"]', list);
        var index = items.indexOf(document.activeElement);
        if (index === -1) return;
        event.preventDefault();
        var next = event.key === "ArrowRight"
          ? (index + 1) % items.length
          : (index - 1 + items.length) % items.length;
        items[next].focus();
        items[next].click();
      });
    });

    dom.on(dom.qs("#applicant-form"), "submit", submitApplication);

    dom.on(dom.qs("#reset-form"), "click", function () {
      form.resetToExample();
      clearFormError();
    });

    dom.on(dom.qs("#form-summary"), "click", function (event) {
      var link = event.target.closest("[data-focus-field]");
      if (!link) return;
      event.preventDefault();
      var target = dom.qs("#f-" + link.getAttribute("data-focus-field"));
      if (target) target.focus();
    });

    var search = dom.qs("#search-input");
    var runSearch = dom.debounce(function () {
      state.filters.q = search.value;
      state.page = 1;
      views.applyFilters();
    }, 180);
    dom.on(search, "input", runSearch);
    dom.on(search, "search", function () {
      state.filters.q = search.value;
      state.page = 1;
      views.applyFilters();
    });

    dom.on(dom.qs("#band-filter"), "change", function (event) {
      state.filters.band = event.target.value;
      state.page = 1;
      views.applyFilters();
    });

    dom.on(dom.qs("#page-size"), "change", function (event) {
      state.pageSize = Number(event.target.value) || 25;
      state.page = 1;
      views.applyFilters();
    });

    var fileControl = dom.qs("#file-control");
    var fileInput = dom.qs("#file-input");

    dom.on(fileControl, "click", function () { fileInput.click(); });
    dom.on(fileControl, "keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });
    dom.on(fileInput, "change", function (event) {
      var file = event.target.files && event.target.files[0];
      scoreBatchFile(file);
      event.target.value = "";
    });

    ["dragenter", "dragover"].forEach(function (name) {
      dom.on(fileControl, name, function (event) {
        event.preventDefault();
        fileControl.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      dom.on(fileControl, name, function (event) {
        event.preventDefault();
        fileControl.classList.remove("is-dragging");
      });
    });
    dom.on(fileControl, "drop", function (event) {
      var files = event.dataTransfer && event.dataTransfer.files;
      if (files && files.length) scoreBatchFile(files[0]);
    });

    dom.qsa("#dataset-list [data-demo]").forEach(function (btn) {
      dom.on(btn, "click", function () {
        loadDemo(btn.getAttribute("data-demo"), btn.getAttribute("data-filename"));
      });
    });

    dom.on(dom.qs("#export-csv"), "click", exportCsv);
    dom.on(dom.qs("#theme-toggle"), "click", toggleTheme);

    dom.on(document, "keydown", function (event) {
      if (event.key === "Escape") closeDrawer();
      trapFocus(event);
    });
  }

  /* ----------------------------------------------------------------------
   * Boot
   * -------------------------------------------------------------------- */
  function renderColumnList() {
    var host = dom.qs("#column-list");
    if (host) host.textContent = form.CSV_COLUMNS.join(", ");
  }

  document.addEventListener("DOMContentLoaded", function () {
    form.renderForm();
    renderColumnList();
    CRC.hydrateIcons(document);
    applyThemeIcon();
    bindEvents();
    setMode("single");
    initHealth();
  });
})(window.CRC);
