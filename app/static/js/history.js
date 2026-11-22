/* PulmoLens — History page. */
(() => {
  const { History, OUTCOMES, toast } = window.PL;
  const grid = document.getElementById("history-grid");
  const empty = document.getElementById("history-empty");
  const noMatch = document.getElementById("history-nomatch");
  const tools = document.querySelector(".history-tools");
  const clearBtn = document.getElementById("clear-history");
  const search = document.getElementById("history-search");
  let filter = "all";

  const fmtDate = (iso) => {
    const d = new Date(iso);
    const today = new Date();
    const sameDay = d.toDateString() === today.toDateString();
    return sameDay
      ? "Today, " + d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
      : d.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
  };
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function render() {
    const all = History.all();
    const has = all.length > 0;
    empty.hidden = has;
    tools.hidden = !has;
    clearBtn.hidden = !has;

    document.querySelectorAll(".chip").forEach((c) => {
      const f = c.dataset.filter;
      const n = f === "all" ? all.length : all.filter((s) => s.result.flat_pred_label === f).length;
      c.querySelector(".chip-count").textContent = n;
      c.classList.toggle("is-on", f === filter);
    });

    const q = search.value.trim().toLowerCase();
    const list = all.filter((s) =>
      (filter === "all" || s.result.flat_pred_label === filter) && (!q || s.name.toLowerCase().includes(q)));
    noMatch.hidden = !has || list.length > 0;

    grid.innerHTML = list.map((s, i) => {
      const r = s.result;
      return `
      <a class="scan-card glass" href="/analyze?scan=${s.id}" style="--i:${i}; --c:${OUTCOMES[r.flat_pred_label] || "var(--cyan)"}" data-id="${s.id}">
        <div class="scan-thumb">
          ${s.thumb ? `<img src="${s.thumb}" alt="">` : ""}
          ${s.thumbHeat ? `<img class="t-heat" src="${s.thumbHeat}" alt="">` : ""}
          <span class="scan-badge"><i></i>${esc(r.flat_pred_label)}</span>
          <button class="icon-btn scan-del" type="button" aria-label="Remove ${esc(s.name)} from history">
            <svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>
          </button>
        </div>
        <div class="scan-info">
          <strong>${esc(s.name)}</strong>
          <span><em style="font-style:normal">${parseFloat(r.confidence).toFixed(0)}% likelihood</em><time>${fmtDate(s.date)}</time></span>
        </div>
      </a>`;
    }).join("");
  }

  document.querySelector(".chips").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    filter = chip.dataset.filter;
    render();
  });
  search.addEventListener("input", render);

  grid.addEventListener("click", (e) => {
    const del = e.target.closest(".scan-del");
    if (!del) return;
    e.preventDefault();
    const card = del.closest(".scan-card");
    card.classList.add("is-removing");
    card.addEventListener("animationend", () => { History.remove(card.dataset.id); render(); toast("Removed from history"); }, { once: true });
  });

  clearBtn.addEventListener("click", () => {
    if (clearBtn.dataset.confirm) {
      History.clear();
      delete clearBtn.dataset.confirm;
      clearBtn.lastChild.textContent = " Clear history";
      render();
      toast("History cleared");
      return;
    }
    clearBtn.dataset.confirm = "1";
    clearBtn.lastChild.textContent = " Select again to clear all";
    setTimeout(() => {
      if (clearBtn.dataset.confirm) { delete clearBtn.dataset.confirm; clearBtn.lastChild.textContent = " Clear history"; }
    }, 3500);
  });

  render();
})();
