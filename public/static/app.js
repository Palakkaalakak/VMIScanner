(function () {
  let DATA = null;
  let FILTERED = [];

  const el = (id) => document.getElementById(id);

  function statusPill(status) {
    return `<span class="status-pill status-${status}">${status}</span>`;
  }

  function fmtDate(iso) {
    if (!iso) return '';
    try {
      const d = new Date(iso);
      return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
    } catch (e) { return iso; }
  }

  async function load() {
    try {
      const res = await fetch('/data/scan_results.json', { cache: 'no-store' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      DATA = await res.json();
    } catch (e) {
      el('results-body').innerHTML =
        `<tr><td colspan="9" class="p-8 text-center text-slate-400">
          No scan results yet. Run the scanner (<code>python3 -m vmi.scan</code>) to generate
          <code>public/data/scan_results.json</code>, then reload.
        </td></tr>`;
      el('meta-info').textContent = 'No data loaded';
      return;
    }
    populateSectorFilter();
    renderSummary();
    applyFilters();
  }

  function populateSectorFilter() {
    const sectors = new Set();
    (DATA.results || []).forEach((r) => { if (r.sector) sectors.add(r.sector); });
    const sel = el('filter-sector');
    Array.from(sectors).sort().forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      sel.appendChild(opt);
    });
  }

  function renderSummary() {
    const c = DATA.counts || {};
    el('meta-info').innerHTML =
      `Universe scanned: ${DATA.universe_size} &middot; Generated ${fmtDate(DATA.generated_at)}`;
    const cards = [
      { label: 'Great Businesses', value: c.great, color: 'text-emerald-600', icon: 'fa-trophy' },
      { label: 'Near Misses', value: c.near_miss, color: 'text-amber-600', icon: 'fa-triangle-exclamation' },
      { label: 'Did Not Qualify', value: c.failed, color: 'text-slate-500', icon: 'fa-xmark' },
      { label: 'Data Errors', value: c.errors, color: 'text-red-500', icon: 'fa-bug' },
    ];
    el('summary-cards').innerHTML = cards.map((card) => `
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
        <div class="flex items-center justify-between">
          <div>
            <div class="text-xs font-semibold text-slate-500 uppercase">${card.label}</div>
            <div class="text-3xl font-bold ${card.color} mt-1">${card.value ?? 0}</div>
          </div>
          <i class="fas ${card.icon} text-2xl ${card.color} opacity-60"></i>
        </div>
      </div>`).join('');
  }

  function applyFilters() {
    if (!DATA) return;
    const q = el('search').value.trim().toLowerCase();
    const status = el('filter-status').value;
    const sector = el('filter-sector').value;
    const ctype = el('filter-type').value;
    const sortBy = el('sort-by').value;

    let rows = (DATA.results || []).filter((r) => !r.error);

    if (status === 'great') rows = rows.filter((r) => r.is_great);
    else if (status === 'near') rows = rows.filter((r) => !r.is_great && r.n_fail <= 1);

    if (sector) rows = rows.filter((r) => r.sector === sector);
    if (ctype) rows = rows.filter((r) => r.company_type === ctype);
    if (q) rows = rows.filter((r) =>
      r.ticker.toLowerCase().includes(q) || (r.company || '').toLowerCase().includes(q));

    if (sortBy === 'ticker') rows = rows.slice().sort((a, b) => a.ticker.localeCompare(b.ticker));
    else rows = rows.slice().sort((a, b) => b.score - a.score || a.n_fail - b.n_fail);

    FILTERED = rows;
    renderTable();
  }

  function renderTable() {
    const tbody = el('results-body');
    const empty = el('empty-state');
    if (!FILTERED.length) {
      tbody.innerHTML = '';
      empty.classList.remove('hidden');
      return;
    }
    empty.classList.add('hidden');
    tbody.innerHTML = FILTERED.map((r, idx) => `
      <tr class="result-row border-b border-slate-100" data-idx="${idx}">
        <td class="px-4 py-2 font-bold text-slate-800">${r.ticker}</td>
        <td class="px-4 py-2 text-slate-600">${r.company || ''}</td>
        <td class="px-4 py-2 text-slate-500 text-xs">${r.sector || ''}<br><span class="text-slate-400">${r.industry || ''}</span></td>
        <td class="px-4 py-2 text-xs">${r.company_type}</td>
        <td class="px-4 py-2 text-center font-bold ${r.is_great ? 'text-emerald-600' : 'text-slate-500'}">${r.score}%</td>
        <td class="px-4 py-2 text-center text-emerald-600">${r.n_pass}</td>
        <td class="px-4 py-2 text-center text-red-500">${r.n_fail}</td>
        <td class="px-4 py-2 text-center text-amber-500">${r.n_warn}</td>
        <td class="px-4 py-2 text-center"><i class="fas fa-chevron-right text-slate-300"></i></td>
      </tr>`).join('');

    tbody.querySelectorAll('tr.result-row').forEach((tr) => {
      tr.addEventListener('click', () => openDetail(FILTERED[Number(tr.dataset.idx)]));
    });
  }

  function openDetail(r) {
    el('modal-title').textContent = `${r.ticker} — ${r.company || ''}`;
    const checksHtml = (r.checks || []).map((c) => `
      <tr class="border-b border-slate-100">
        <td class="py-2 pr-3 text-sm text-slate-700">${c.name}</td>
        <td class="py-2 pr-3">${statusPill(c.status)}</td>
        <td class="py-2 pr-3 text-sm text-slate-500">${c.value || ''}</td>
        <td class="py-2 text-xs text-slate-400">${c.detail || ''}</td>
      </tr>`).join('');

    const hints = r.moat_hints || {};
    const hintsHtml = Object.keys(hints)
      .filter((k) => k !== 'verdict')
      .map((k) => `<li><strong>${k.replace(/_/g, ' ')}:</strong> ${hints[k]}</li>`).join('');

    el('modal-body').innerHTML = `
      <div class="mb-4 flex gap-4 text-sm text-slate-500">
        <span>${r.sector || ''} / ${r.industry || ''}</span>
        <span>&middot;</span>
        <span>${r.market_cap ? 'Mkt cap ' + r.market_cap : ''}</span>
        <span>&middot;</span>
        <span>Type: ${r.company_type}</span>
      </div>
      <table class="w-full mb-5"><tbody>${checksHtml}</tbody></table>
      <div class="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <h4 class="font-bold text-amber-800 mb-2 text-sm"><i class="fas fa-shield-halved mr-1"></i>Moat hints — your judgment call</h4>
        <ul class="text-sm text-amber-900 space-y-1 list-disc pl-5">${hintsHtml}</ul>
        <p class="text-xs text-amber-700 mt-2 italic">${hints.verdict || ''}</p>
      </div>`;
    el('detail-modal').classList.remove('hidden');
  }

  el('modal-close').addEventListener('click', () => el('detail-modal').classList.add('hidden'));
  el('detail-modal').addEventListener('click', (e) => {
    if (e.target.id === 'detail-modal') el('detail-modal').classList.add('hidden');
  });
  ['search', 'filter-status', 'filter-sector', 'filter-type', 'sort-by'].forEach((id) => {
    el(id).addEventListener('input', applyFilters);
    el(id).addEventListener('change', applyFilters);
  });

  load();
})();
