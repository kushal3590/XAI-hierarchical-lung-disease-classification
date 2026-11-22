/* PulmoLens — shared behaviour: nav, reveals, toasts, local scan history. */
(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Top bar: scrolled state ---------- */
  const onScroll = () => document.body.classList.toggle("is-scrolled", window.scrollY > 8);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  /* ---------- Nav pill that glides between links ---------- */
  const nav = document.querySelector(".nav");
  if (nav) {
    const pill = nav.querySelector(".nav-pill");
    const links = [...nav.querySelectorAll(".nav-link")];
    const active = nav.querySelector(".nav-link.is-active");
    const moveTo = (el) => {
      if (!el) { pill.style.opacity = 0; return; }
      pill.style.opacity = 1;
      pill.style.width = el.offsetWidth + "px";
      pill.style.transform = `translateX(${el.offsetLeft}px)`;
    };
    pill.style.transition = "none";
    moveTo(active);
    requestAnimationFrame(() => (pill.style.transition = ""));
    links.forEach((l) => l.addEventListener("mouseenter", () => moveTo(l)));
    nav.addEventListener("mouseleave", () => moveTo(active));
    window.addEventListener("resize", () => moveTo(active));
    document.fonts && document.fonts.ready.then(() => moveTo(active));
  }

  /* ---------- Mobile menu ---------- */
  const menuBtn = document.querySelector(".menu-btn");
  const mobileNav = document.getElementById("mobile-nav");
  if (menuBtn && mobileNav) {
    menuBtn.addEventListener("click", () => {
      const open = menuBtn.getAttribute("aria-expanded") === "true";
      menuBtn.setAttribute("aria-expanded", String(!open));
      mobileNav.hidden = open;
    });
  }

  /* ---------- Hero headline: word-by-word rise ---------- */
  document.querySelectorAll("[data-split]").forEach((el) => {
    const words = el.textContent.trim().split(/\s+/);
    el.setAttribute("aria-label", el.textContent.trim());
    el.innerHTML = words
      .map((w, i) => `<span class="w" aria-hidden="true"><span style="--i:${i}">${w}</span></span>`)
      .join(" ");
  });

  /* ---------- Scroll reveals ---------- */
  document.querySelectorAll(".steps .step").forEach((s, i) => s.querySelector(".step-num").style.setProperty("--i", i));
  const revealEls = document.querySelectorAll("[data-reveal], .doc-section");
  if ("IntersectionObserver" in window && !reduceMotion) {
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("is-in"); io.unobserve(e.target); }
      }),
      { threshold: 0.18, rootMargin: "0px 0px -40px 0px" }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("is-in"));
  }

  /* ---------- Soft spotlight that follows the cursor on glass panels ---------- */
  if (!reduceMotion && window.matchMedia("(hover: hover)").matches) {
    document.addEventListener("pointermove", (e) => {
      const el = e.target.closest && e.target.closest("[data-spot]");
      if (!el) return;
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${e.clientX - r.left}px`);
      el.style.setProperty("--my", `${e.clientY - r.top}px`);
    }, { passive: true });
  }

  /* ---------- Range inputs: filled track ---------- */
  const paintRange = (r) => {
    const p = ((r.value - r.min) / (r.max - r.min)) * 100;
    r.style.setProperty("--p", p + "%");
  };
  document.querySelectorAll('input[type="range"]').forEach((r) => {
    paintRange(r);
    r.addEventListener("input", () => paintRange(r));
  });

  /* ---------- Toasts ---------- */
  const stack = document.querySelector(".toast-stack");
  function toast(message, kind = "info") {
    if (!stack) return;
    const t = document.createElement("div");
    t.className = "toast" + (kind === "error" ? " is-error" : "");
    t.innerHTML = "<i></i><span></span>";
    t.querySelector("span").textContent = message;
    stack.appendChild(t);
    setTimeout(() => {
      t.classList.add("is-leaving");
      t.addEventListener("animationend", () => t.remove(), { once: true });
    }, 3600);
  }

  /* ---------- Local scan history (browser only) ---------- */
  const KEY = "pulmolens.history.v1";
  const MAX = 24;
  const History = {
    all() {
      try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; }
    },
    get(id) { return this.all().find((s) => s.id === id); },
    save(entry) {
      let list = [entry, ...this.all().filter((s) => s.id !== entry.id)].slice(0, MAX);
      while (list.length) {
        try { localStorage.setItem(KEY, JSON.stringify(list)); return true; }
        catch { list = list.slice(0, -1); } // storage full: drop the oldest and retry
      }
      return false;
    },
    remove(id) {
      try { localStorage.setItem(KEY, JSON.stringify(this.all().filter((s) => s.id !== id))); } catch {}
    },
    clear() { try { localStorage.removeItem(KEY); } catch {} },
  };

  const OUTCOMES = {
    "Normal": "var(--normal)",
    "Pneumonia": "var(--pneumonia)",
    "Tuberculosis": "var(--tb)",
    "COVID-19": "var(--covid)",
  };

  window.PL = { toast, History, OUTCOMES, paintRange, reduceMotion };
})();
