/* PulmoLens — Analyze page. */
(() => {
  const { toast, History, OUTCOMES, paintRange, reduceMotion } = window.PL;
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];

  const workspace = $(".workspace");
  const DEFAULT_THRESHOLD = Number(workspace.dataset.defaultThreshold || 25);

  const form = $("#scan-form");
  const fileInput = $("#xray");
  const dropzone = $("#dropzone");
  const runBtn = $("#run-btn");
  const threshold = $("#threshold");
  const thresholdOut = $("#threshold-out");
  const shapEvals = $("#shap_evals");
  const shapOut = $("#shap-out");
  const shapField = $("#shap-field");

  let currentFile = null;
  let currentPreview = null;
  let lastResult = null;
  let stepTimers = [];

  /* ======================= States ======================= */
  function setState(name) {
    $$(".state").forEach((s) => s.classList.toggle("is-visible", s.dataset.state === name));
    const top = $(".stage").getBoundingClientRect().top;
    if (name !== "empty" && (window.innerWidth < 1080 || top < 60)) {
      $(".stage").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    }
  }

  /* ======================= File picking ======================= */
  const fmtSize = (b) => (b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1e3)) + " KB");

  function pickFile(file) {
    if (!file) return;
    if (!/^image\/(png|jpe?g)$/.test(file.type)) {
      toast("That file isn't a JPG or PNG image. Choose a chest X-ray image.", "error");
      return;
    }
    currentFile = file;
    if (currentPreview) URL.revokeObjectURL(currentPreview);
    currentPreview = URL.createObjectURL(file);
    $(".dz-thumb").src = currentPreview;
    $(".dz-name").textContent = file.name;
    $(".dz-size").textContent = fmtSize(file.size);
    $(".dz-empty").hidden = true;
    $(".dz-file").hidden = false;
    runBtn.disabled = false;
  }

  function clearFile() {
    currentFile = null;
    fileInput.value = "";
    $(".dz-empty").hidden = false;
    $(".dz-file").hidden = true;
    runBtn.disabled = true;
  }

  fileInput.addEventListener("change", () => pickFile(fileInput.files[0]));
  $(".dz-clear").addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); clearFile(); });
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("is-over"); }));
  dropzone.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
  // Allow dropping anywhere on the page
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]); });
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) pickFile(item.getAsFile());
  });

  /* ======================= Options ======================= */
  const isDeep = () => form.depth.value === "deep";
  const syncDepth = () => shapField.classList.toggle("is-dim", !isDeep());
  $$('input[name="depth"]').forEach((r) => r.addEventListener("change", syncDepth));
  syncDepth();

  threshold.addEventListener("input", () => (thresholdOut.textContent = threshold.value + "%"));
  shapEvals.addEventListener("input", () => (shapOut.textContent = shapEvals.value));
  $("#threshold-reset").addEventListener("click", () => {
    threshold.value = DEFAULT_THRESHOLD;
    thresholdOut.textContent = DEFAULT_THRESHOLD + "%";
    paintRange(threshold);
  });

  /* ======================= Loading sequence ======================= */
  function startSteps(deep) {
    const stage = $(".state-loading");
    stage.classList.toggle("is-deep", deep);
    const steps = $$(".scan-steps li").filter((li) => deep || !li.classList.contains("deep-only"));
    steps.forEach((li) => li.classList.remove("is-active", "is-done"));
    const timings = [0, 700, 1600, 2500, 3600];
    steps.forEach((li, i) => {
      stepTimers.push(setTimeout(() => {
        steps.slice(0, i).forEach((p) => { p.classList.remove("is-active"); p.classList.add("is-done"); });
        li.classList.add("is-active");
      }, timings[i]));
    });
    return steps;
  }

  function finishSteps(steps) {
    stepTimers.forEach(clearTimeout);
    stepTimers = [];
    return new Promise((resolve) => {
      let i = 0;
      const tick = () => {
        if (i >= steps.length) return setTimeout(resolve, 250);
        steps[i].classList.remove("is-active");
        steps[i].classList.add("is-done");
        i++;
        setTimeout(tick, reduceMotion ? 0 : 140);
      };
      tick();
    });
  }

  /* ======================= Submit ======================= */
  async function run() {
    if (!currentFile) return;
    const deep = isDeep();
    $(".scan-img").src = currentPreview;
    setState("loading");
    runBtn.classList.add("is-busy");
    runBtn.disabled = true;
    $(".btn-label", runBtn).textContent = "Analyzing…";
    const steps = startSteps(deep);
    const started = performance.now();

    const body = new FormData();
    body.append("xray", currentFile);
    body.append("threshold_pct", threshold.value);
    body.append("deep", deep ? "1" : "0");
    body.append("shap_evals", shapEvals.value);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      let data;
      try { data = await res.json(); } catch { throw new Error(`The server returned an unexpected response (HTTP ${res.status}).`); }
      if (!res.ok || data.error) throw new Error(data.error || `Request failed (HTTP ${res.status}).`);

      // Keep the scan animation on screen long enough to read
      const minTime = 1800 - (performance.now() - started);
      if (minTime > 0) await new Promise((r) => setTimeout(r, minTime));
      await finishSteps(steps);

      const entry = {
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
        name: currentFile.name,
        date: new Date().toISOString(),
        threshold: Number(threshold.value),
        result: data,
      };
      renderResult(entry, true);
      persist(entry);
    } catch (err) {
      stepTimers.forEach(clearTimeout);
      const msg = err instanceof TypeError
        ? "Couldn't reach the PulmoLens server. Check that app_flask.py is still running, then try again."
        : err.message;
      $(".error-msg").textContent = msg;
      setState("error");
    } finally {
      runBtn.classList.remove("is-busy");
      runBtn.disabled = !currentFile;
      $(".btn-label", runBtn).textContent = "Analyze X-ray";
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $('[data-action="retry"]').addEventListener("click", () => (currentFile ? run() : setState("empty")));

  /* ======================= Result rendering ======================= */

  let layers = {};
  let currentLayer = "gradcam1";
  let captions = {};

  function renderResult(entry, fresh) {
    lastResult = entry;
    const r = entry.result;
    const label = r.flat_pred_label;
    const color = OUTCOMES[label] || "var(--cyan)";
    const conf = parseFloat(r.confidence);
    const abnormal = r.stage1_label === "Abnormal";
    const stage2Used = abnormal && r.gradcam2_b64 && !/^N\/A/.test(r.stage2_text);
    const thr = r.stage1_threshold_pct ?? entry.threshold ?? DEFAULT_THRESHOLD;

    const root = $(".state-result");
    root.style.setProperty("--c", color);
    $(".verdict").style.setProperty("--c", color);
    $(".verdict-label").textContent = label;
    $(".verdict-sub").textContent = `From ${entry.name}`;
    $(".saved-note").hidden = fresh;

    // Print header
    let ph = $(".print-head", root);
    if (!ph) { ph = document.createElement("div"); ph.className = "print-head"; root.prepend(ph); }
    ph.textContent = `PulmoLens report · ${entry.name} · ${new Date(entry.date).toLocaleString()} · Research use only, not a diagnosis.`;

    setState("result");

    // Confidence ring + count-up
    const ring = $(".ring-fill");
    const C = 2 * Math.PI * 52;
    ring.style.transition = "none";
    ring.style.strokeDashoffset = C;
    ring.getBoundingClientRect();
    ring.style.transition = "";
    requestAnimationFrame(() => (ring.style.strokeDashoffset = C * (1 - conf / 100)));
    countUp($(".ring-num"), Math.round(conf));

    // Decision path
    const n1 = $('[data-node="1"]');
    const n2 = $('[data-node="2"]');
    [n1, n2].forEach((n) => n.classList.remove("is-lit", "is-flowing", "is-skipped"));
    $(".stage1-value").textContent = abnormal ? "Flagged abnormal" : "Looks normal";
    const score = r.stage1_abnormal_pct;
    $(".stage1-hint").textContent = score != null
      ? `Screening score ${score}%, cutoff ${thr}%.`
      : `Cutoff ${thr}%.`;
    $(".meter-fill").style.width = "0";
    $(".meter-cut").style.left = thr + "%";
    if (stage2Used) {
      $(".stage2-value").textContent = r.stage2_text;
      $(".stage2-hint").textContent = "Chosen from pneumonia, tuberculosis and COVID-19.";
    } else {
      $(".stage2-value").textContent = "Not needed";
      $(".stage2-hint").textContent = "Only abnormal scans go on to this step.";
    }
    setTimeout(() => {
      n1.classList.add("is-lit");
      $(".meter-fill").style.width = (score ?? (abnormal ? 100 : 0)) + "%";
      $(".meter-fill").style.background = abnormal ? "var(--warn)" : "var(--ok)";
    }, 250);
    setTimeout(() => {
      if (stage2Used) { n1.classList.add("is-flowing"); setTimeout(() => n2.classList.add("is-lit"), 650); }
      else n2.classList.add("is-skipped");
    }, 700);

    // Probability bars
    const bars = $(".bars");
    bars.innerHTML = "";
    const top = Math.max(...r.probs_list.map((p) => p.pct));
    r.probs_list.forEach((p) => {
      const li = document.createElement("li");
      li.className = "bar" + (p.pct === top ? " is-top" : "");
      li.style.setProperty("--c", OUTCOMES[p.label] || "var(--cyan)");
      li.innerHTML = `<span class="bar-name"><i></i>${p.label}</span><span class="bar-pct">${p.pct.toFixed(1)}%</span><span class="bar-track"><span class="bar-fill"></span></span>`;
      bars.appendChild(li);
    });
    setTimeout(() => $$(".bar-fill", bars).forEach((f, i) => {
      setTimeout(() => (f.style.width = r.probs_list[i].pct + "%"), i * 90);
    }), 350);


    // Viewer layers
    layers = {
      input: r.input_img_b64,
      gradcam1: r.gradcam1_b64,
      gradcam2: stage2Used ? r.gradcam2_b64 : null,
      shap: r.shap_b64 || null,
    };
    captions = {
      input: "The X-ray after contrast enhancement, exactly as the model saw it.",
      gradcam1: `Where screening looked when deciding the scan ${abnormal ? "was abnormal" : "was normal"}. Drag across the image to compare with the original.`,
      gradcam2: `Where classification looked when choosing ${r.stage2_text}. Drag across the image to compare with the original.`,
      shap: "Red areas pushed the model toward its answer; blue areas pushed it away.",
    };
    const tabs = $$(".viewer-tabs button");
    tabs.forEach((t) => {
      const key = t.dataset.layer;
      if (key === "shap") t.hidden = !layers.shap;
      t.disabled = !layers[key];
      t.title = !layers[key] && key === "gradcam2" ? "Not created: the scan passed screening as normal" : "";
    });
    $(".viewer-base").src = src(layers.input);
    selectLayer(fresh && layers.shap ? "gradcam1" : "gradcam1", true);
    if (fresh) hintCompare();
  }

  const src = (b64) => (b64 && b64.startsWith("data:") ? b64 : "data:image/png;base64," + b64);

  function countUp(el, to) {
    if (reduceMotion) { el.textContent = to; return; }
    const start = performance.now();
    const dur = 1300;
    const step = (now) => {
      const t = Math.min(1, (now - start) / dur);
      el.textContent = Math.round(to * (1 - Math.pow(1 - t, 4)));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ---------- Layer tabs ---------- */
  const tabPill = $(".tab-pill");
  function placePill(btn, instant) {
    if (instant) tabPill.style.transition = "none";
    tabPill.style.width = btn.offsetWidth + "px";
    tabPill.style.transform = `translateX(${btn.offsetLeft - 4}px)`;
    if (instant) requestAnimationFrame(() => (tabPill.style.transition = ""));
  }

  function selectLayer(key, instant) {
    if (!layers[key]) return;
    currentLayer = key;
    const btn = $(`.viewer-tabs [data-layer="${key}"]`);
    $$(".viewer-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    requestAnimationFrame(() => placePill(btn, instant));
    const viewer = $("#viewer");
    viewer.classList.toggle("is-original", key === "input");
    const img = $(".viewer-overlay img");
    if (key !== "input") {
      if (instant) img.src = src(layers[key]);
      else {
        img.classList.add("is-swapping");
        setTimeout(() => { img.src = src(layers[key]); img.classList.remove("is-swapping"); }, 220);
      }
    }
    $(".layer-caption").textContent = captions[key];
  }

  $$(".viewer-tabs button").forEach((b) => b.addEventListener("click", () => selectLayer(b.dataset.layer)));
  $(".viewer-tabs").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const tabs = $$(".viewer-tabs button").filter((b) => !b.hidden && !b.disabled);
    const i = tabs.findIndex((b) => b.dataset.layer === currentLayer);
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    selectLayer(next.dataset.layer);
    next.focus();
  });
  window.addEventListener("resize", () => {
    const btn = $(`.viewer-tabs [data-layer="${currentLayer}"]`);
    if (btn && btn.offsetWidth) placePill(btn, true);
  });

  /* ---------- Compare slider ---------- */
  const range = $(".compare-range");
  const overlay = $(".viewer-overlay");
  const handle = $(".compare-handle");
  function setCompare(v) {
    overlay.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
    handle.style.left = v + "%";
    handle.style.opacity = v >= 99.5 ? 0 : 1;
  }
  range.addEventListener("input", () => setCompare(Number(range.value)));
  range.addEventListener("dblclick", () => { range.value = 100; setCompare(100); });
  setCompare(100);

  function hintCompare() {
    if (reduceMotion) return;
    const start = performance.now() + 900;
    const dur = 1800;
    const ease = (t) => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const frame = (now) => {
      const t = (now - start) / dur;
      if (t < 0) return requestAnimationFrame(frame);
      if (t >= 1) { range.value = 100; setCompare(100); return; }
      const v = 100 - 45 * Math.sin(Math.PI * ease(t));
      range.value = v;
      setCompare(v);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  /* ---------- Full screen ---------- */
  const modal = $("#lightbox-modal");
  $(".viewer-expand").addEventListener("click", () => {
    $("img", modal).src = src(layers[currentLayer]);
    modal.hidden = false;
    $(".modal-close", modal).focus();
  });
  const closeModal = () => (modal.hidden = true);
  modal.addEventListener("click", (e) => { if (e.target === modal || e.target.closest(".modal-close")) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !modal.hidden) closeModal(); });

  /* ---------- Actions ---------- */
  $('[data-action="print"]').addEventListener("click", () => window.print());
  $('[data-action="new"]').addEventListener("click", () => {
    clearFile();
    history.replaceState(null, "", "/analyze");
    setState("empty");
    window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
  });

  /* ======================= History persistence ======================= */
  function shrink(b64, max, quality = 0.82) {
    return new Promise((resolve) => {
      if (!b64) return resolve(null);
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, max / Math.max(img.width, img.height));
        const c = document.createElement("canvas");
        c.width = Math.round(img.width * scale);
        c.height = Math.round(img.height * scale);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL("image/jpeg", quality));
      };
      img.onerror = () => resolve(null);
      img.src = src(b64);
    });
  }

  async function persist(entry) {
    const r = entry.result;
    const [input, g1, g2, shap, thumb, thumbHeat] = await Promise.all([
      shrink(r.input_img_b64, 448),
      shrink(r.gradcam1_b64, 448),
      shrink(r.gradcam2_b64, 448),
      shrink(r.shap_b64, 520),
      shrink(r.input_img_b64, 220, 0.75),
      shrink(r.gradcam1_b64, 220, 0.75),
    ]);
    const stored = {
      ...entry,
      thumb, thumbHeat,
      result: { ...r, input_img_b64: input, gradcam1_b64: g1, gradcam2_b64: g2, shap_b64: shap },
    };
    if (History.save(stored)) {
      history.replaceState(null, "", "/analyze?scan=" + entry.id);
      const note = $(".saved-note");
      note.hidden = false;
      note.classList.remove("is-in");
      requestAnimationFrame(() => note.classList.add("is-in"));
    }
  }

  /* ======================= Open a saved scan ======================= */
  const scanId = new URLSearchParams(location.search).get("scan");
  if (scanId) {
    const saved = History.get(scanId);
    if (saved) renderResult(saved, false);
    else toast("That scan is no longer in your history.", "error");
  }
})();
