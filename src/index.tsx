import { Hono } from 'hono'
import { serveStatic } from 'hono/cloudflare-workers'

const app = new Hono()

app.use('/static/*', serveStatic({ root: './public' }))
app.use('/data/*', serveStatic({ root: './public' }))

app.get('/api/health', (c) => c.json({ ok: true }))

app.get('/', (c) => {
  return c.html(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VMI Great Business Scanner</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
<link href="/static/styles.css" rel="stylesheet">
</head>
<body class="bg-slate-50 text-slate-800">
<header class="bg-slate-900 text-white px-6 py-5 sticky top-0 z-20 shadow">
  <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
    <div>
      <h1 class="text-2xl font-bold"><i class="fas fa-magnifying-glass-chart mr-2 text-emerald-400"></i>VMI Great Business Scanner</h1>
      <p class="text-slate-300 text-sm mt-1" id="subtitle">Value Momentum Investing — fundamentals-only screen (Adam Khoo / Piranha Profits methodology)</p>
    </div>
    <div class="text-right text-xs text-slate-400" id="meta-info"></div>
  </div>
</header>

<main class="max-w-7xl mx-auto px-6 py-6">

  <section id="summary-cards" class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6"></section>

  <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-4 mb-6">
    <div class="flex flex-wrap gap-3 items-end">
      <div class="flex-1 min-w-[220px]">
        <label class="block text-xs font-semibold text-slate-500 mb-1">SEARCH TICKER / COMPANY</label>
        <input id="search" type="text" placeholder="e.g. AAPL, Microsoft..." class="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400">
      </div>
      <div>
        <label class="block text-xs font-semibold text-slate-500 mb-1">RESULT</label>
        <select id="filter-status" class="border border-slate-300 rounded-lg px-3 py-2 text-sm">
          <option value="great">Great Businesses (0 fails)</option>
          <option value="near">Near Misses (1 fail)</option>
          <option value="all">All Scanned</option>
        </select>
      </div>
      <div>
        <label class="block text-xs font-semibold text-slate-500 mb-1">SECTOR</label>
        <select id="filter-sector" class="border border-slate-300 rounded-lg px-3 py-2 text-sm">
          <option value="">All sectors</option>
        </select>
      </div>
      <div>
        <label class="block text-xs font-semibold text-slate-500 mb-1">COMPANY TYPE</label>
        <select id="filter-type" class="border border-slate-300 rounded-lg px-3 py-2 text-sm">
          <option value="">All types</option>
          <option value="standard">Standard</option>
          <option value="financial">Financial / Bank</option>
          <option value="reit">REIT</option>
          <option value="property">Property developer</option>
          <option value="commodity">Commodity</option>
        </select>
      </div>
      <div>
        <label class="block text-xs font-semibold text-slate-500 mb-1">SORT</label>
        <select id="sort-by" class="border border-slate-300 rounded-lg px-3 py-2 text-sm">
          <option value="score">Score (high to low)</option>
          <option value="ticker">Ticker (A-Z)</option>
        </select>
      </div>
    </div>
  </section>

  <section class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead class="bg-slate-100 text-slate-600 text-xs uppercase">
          <tr>
            <th class="px-4 py-3 text-left">Ticker</th>
            <th class="px-4 py-3 text-left">Company</th>
            <th class="px-4 py-3 text-left">Sector</th>
            <th class="px-4 py-3 text-left">Type</th>
            <th class="px-4 py-3 text-center">Score</th>
            <th class="px-4 py-3 text-center">Pass</th>
            <th class="px-4 py-3 text-center">Fail</th>
            <th class="px-4 py-3 text-center">Warn</th>
            <th class="px-4 py-3 text-center">Details</th>
          </tr>
        </thead>
        <tbody id="results-body"></tbody>
      </table>
    </div>
    <div id="empty-state" class="hidden p-10 text-center text-slate-400">
      <i class="fas fa-inbox text-3xl mb-2"></i>
      <p>No results match your filters.</p>
    </div>
  </section>

  <section class="mt-8 bg-white rounded-xl shadow-sm border border-slate-200 p-5 text-sm text-slate-600">
    <h2 class="font-bold text-slate-800 mb-2"><i class="fas fa-circle-info mr-1 text-emerald-500"></i>About this scan</h2>
    <p class="mb-2">This scanner pre-filters the universe on Finviz using the VMI-recommended screen
    (positive 5y sales growth, positive EPS growth past/this/next year and next 5 years, ROE &gt; 10%, current ratio &gt; 1),
    then runs a deep 12-13 point fundamental check per ticker against 10 years of financial statements
    from stockanalysis.com — covering profitability, financial strength and management-effectiveness metrics
    exactly as taught in Lessons 4 &amp; 7 of the course. <strong>Valuation (intrinsic value) and technical analysis
    are intentionally excluded</strong> — this tool only answers "is it a great business?", not "is it a great price/entry?".</p>
    <p class="mb-2"><strong>Moat is never auto-scored.</strong> The course explicitly treats sustainable competitive
    advantage as a qualitative judgment — a computed "wide moat score" would misrepresent the framework. Instead each
    result shows moat <em>hints</em> (margin levels, ROIC persistence, buyback yield) so you can apply your own judgment
    using the 5 sources of moat: brand monopoly, high switching costs, network effect, high barriers to entry, economies of scale.</p>
    <p class="text-xs text-slate-400">Banks/financials: CET1 and NPL ratios aren't available from free sources — check the
    company's investor relations / Basel III Pillar 3 disclosures manually. REIT gearing uses Total Debt / Total Assets &lt; 45%.</p>
  </section>
</main>

<div id="detail-modal" class="hidden fixed inset-0 bg-black/50 z-30 flex items-center justify-center p-4">
  <div class="bg-white rounded-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto shadow-2xl">
    <div class="sticky top-0 bg-white border-b border-slate-200 px-6 py-4 flex justify-between items-center">
      <h3 class="text-lg font-bold" id="modal-title"></h3>
      <button id="modal-close" class="text-slate-400 hover:text-slate-700"><i class="fas fa-times text-xl"></i></button>
    </div>
    <div class="px-6 py-4" id="modal-body"></div>
  </div>
</div>

<script src="/static/app.js"></script>
</body>
</html>`)
})

export default app
