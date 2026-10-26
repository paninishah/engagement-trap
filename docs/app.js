"use strict";

// Everything shown on this page comes from the JSON files the Kaggle notebooks exported.
const DATA = "data/";
const K = 10;          // films per list
const POOL = 100;      // candidates the re-rankers may reorder (same as notebook 03)
const EXAMPLE_PICKS = [
  "Toy Story (1995)", "Spirited Away", "Amelie", "Eternal Sunshine of the Spotless Mind",
  "Grand Budapest Hotel, The",
];

// Chart colours (checked for colour-blind separation): one fixed colour per policy.
const POLICIES = [
  { key: "svd", name: "Plain SVD", color: "#8F4E22" },
  { key: "mmr", name: "MMR re-ranking", color: "#CF9800" },
  { key: "kmeans_cap", name: "K-Means cap", color: "#2E78B8" },
  { key: "dbscan_cap", name: "DBSCAN cap", color: "#B24A6E" },
];
const SIM_METRICS = {
  CumCoverage: { label: "Share of catalogue ever shown", fmt: pct },
  TrueUtility: { label: "Hidden P(like) of shown films", fmt: pct },
  Discovery: { label: "Share of each user's 50 best matches ever shown", fmt: pct },
  ExposureGini: { label: "Exposure inequality (Gini)", fmt: (v) => v.toFixed(3) },
  AvgPop: { label: "Average popularity of shown films", fmt: (v) => v.toFixed(0) },
  ILD: { label: "Genre diversity inside lists (ILD)", fmt: (v) => v.toFixed(3) },
  CTR: { label: "Click rate", fmt: pct },
};
const INK = "#2a1a10", MUTED = "#8a7565", GRID = "#efe7dc";

const state = {
  movies: [], k: 0, Q: null, Qn: null, kmeans: [],
  picks: [], method: "mmr", lam: 0.7, cap: 3,
};

function pct(v, digits = 1) { return (100 * v).toFixed(digits) + "%"; }
function $(id) { return document.getElementById(id); }
function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
async function loadJSON(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}

// ------------------------------------------------------------------ maths

// Ridge fold-in: find the user vector p that best explains "likes all picked
// films" with the SVD film vectors Q:  p = (Q_J^T Q_J + reg I)^-1 Q_J^T 1
function foldIn(picks) {
  const { k, Q } = state;
  const A = Array.from({ length: k }, () => new Float64Array(k));
  const b = new Float64Array(k);
  for (const i of picks) {
    const off = i * k;
    for (let a = 0; a < k; a++) {
      const qa = Q[off + a];
      b[a] += qa;
      for (let c = 0; c < k; c++) A[a][c] += qa * Q[off + c];
    }
  }
  let trace = 0;
  for (let a = 0; a < k; a++) trace += A[a][a];
  const reg = 0.1 * trace / k + 1e-6;  // ridge strength relative to the data's scale
  for (let a = 0; a < k; a++) A[a][a] += reg;
  return solve(A, b);
}

// Gaussian elimination with partial pivoting (A is k x k, small).
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
  }
  return x;
}

function scoreAll(p) {
  const { k, Q } = state;
  const n = state.movies.length;
  const s = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let d = 0;
    for (let a = 0; a < k; a++) d += Q[i * k + a] * p[a];
    s[i] = d;
  }
  return s;
}

function cosine(i, j) {
  const { k, Qn } = state;
  let d = 0;
  for (let a = 0; a < k; a++) d += Qn[i * k + a] * Qn[j * k + a];
  return d;
}

// Maximal Marginal Relevance, as in etlib.mmr
function mmr(pool, scores, lam) {
  const rel = pool.map((i) => scores[i]);
  const lo = Math.min(...rel), hi = Math.max(...rel);
  const relN = rel.map((r) => (r - lo) / (hi - lo + 1e-12));
  const maxSim = new Float64Array(pool.length);
  const taken = new Array(pool.length).fill(false);
  const out = [];
  for (let step = 0; step < K; step++) {
    let best = -1, bestVal = -Infinity;
    for (let j = 0; j < pool.length; j++) {
      if (taken[j]) continue;
      const v = lam * relN[j] - (1 - lam) * maxSim[j];
      if (v > bestVal) { bestVal = v; best = j; }
    }
    taken[best] = true;
    out.push(pool[best]);
    for (let j = 0; j < pool.length; j++) maxSim[j] = Math.max(maxSim[j], cosine(pool[best], pool[j]));
  }
  return out;
}

// Cluster cap, as in etlib.cluster_cap
function clusterCap(pool, labels, cap) {
  const out = [], counts = new Map();
  for (const i of pool) {
    const c = labels[i];
    if (c === -1 || (counts.get(c) || 0) < cap) {
      out.push(i);
      counts.set(c, (counts.get(c) || 0) + 1);
      if (out.length === K) return out;
    }
  }
  for (const i of pool) if (out.length < K && !out.includes(i)) out.push(i);
  return out;
}

// ------------------------------------------------------------------ recommender UI

function recommend() {
  const has = state.picks.length > 0;
  $("empty-state").hidden = has;
  $("lists").hidden = !has;
  if (!has) return;

  const scores = scoreAll(foldIn(state.picks));
  const picked = new Set(state.picks);
  const pool = [...scores.keys()].filter((i) => !picked.has(i))
    .sort((a, b) => scores[b] - scores[a]).slice(0, POOL);

  const left = pool.slice(0, K);
  const right = state.method === "mmr" ? mmr(pool, scores, state.lam) : clusterCap(pool, state.kmeans, state.cap);
  const leftSet = new Set(left);

  renderList($("list-left"), left, null);
  renderList($("list-right"), right, leftSet);
  const base = mean(left.map((i) => scores[i]));
  renderStats($("stats-left"), left, scores, base);
  renderStats($("stats-right"), right, scores, base);
  $("right-sub").textContent = state.method === "mmr"
    ? `MMR, λ = ${state.lam.toFixed(2)}` : `K-Means cap, max ${state.cap} per group`;
}

function mean(xs) { return xs.reduce((a, b) => a + b, 0) / xs.length; }

function renderList(ol, ids, compareTo) {
  ol.replaceChildren(...ids.map((i, r) => {
    const m = state.movies[i];
    const li = el("li");
    li.append(el("span", { class: "rank" }, String(r + 1)));
    const t = el("span", { class: "m-title" }, m.title);
    if (compareTo && !compareTo.has(i)) t.append(el("span", { class: "new-tag" }, "new"));
    t.append(el("span", { class: "m-meta" }, m.genres.join(" · ")));
    li.append(t, el("span", { class: "m-pop", title: "number of ratings" }, `${m.n_ratings} ratings`));
    return li;
  }));
}

function renderStats(dl, ids, scores, base) {
  const groups = new Set(ids.map((i) => state.kmeans[i])).size;
  const genres = new Set(ids.flatMap((i) => state.movies[i].genres)).size;
  const appeal = base > 0 ? pct(mean(ids.map((i) => scores[i])) / base, 0) : "—";
  const items = [
    ["Avg popularity", mean(ids.map((i) => state.movies[i].n_ratings)).toFixed(0)],
    ["Taste groups", `${groups} / 10`],
    ["Genres", String(genres)],
    ["Model score", appeal],
  ];
  dl.replaceChildren(...items.map(([k, v]) => {
    const d = el("div");
    d.append(el("dt", {}, k), el("dd", {}, v));
    return d;
  }));
}

function renderPicks() {
  const ul = $("picks");
  ul.replaceChildren(...state.picks.map((i) => {
    const li = el("li", {}, state.movies[i].title);
    const x = el("button", { type: "button", "aria-label": `Remove ${state.movies[i].title}` }, "×");
    x.addEventListener("click", () => { state.picks = state.picks.filter((j) => j !== i); update(); });
    li.append(x);
    return li;
  }));
}

function update() { renderPicks(); recommend(); }

function normalise(s) {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

function setupSearch() {
  const input = $("search-input"), list = $("search-results");
  let matches = [], hl = 0;
  const close = () => { list.hidden = true; };
  const choose = (i) => {
    if (!state.picks.includes(i)) state.picks.push(i);
    input.value = "";
    close();
    update();
    input.focus();
  };
  const draw = () => {
    if (!matches.length) {
      list.replaceChildren(el("li", { class: "r-none" }, "No films match. Only films with 5+ ratings are included."));
    } else {
      list.replaceChildren(...matches.map((i, n) => {
        const m = state.movies[i];
        const li = el("li", { role: "option", class: n === hl ? "hl" : "" });
        li.append(el("div", { class: "r-title" }, m.title),
          el("div", { class: "r-meta" }, `${m.genres.join(" · ")} · ${m.n_ratings} ratings`));
        li.addEventListener("mousedown", (e) => { e.preventDefault(); choose(i); });
        return li;
      }));
    }
    list.hidden = false;
  };
  input.addEventListener("input", () => {
    const words = normalise(input.value).split(/\s+/).filter(Boolean);
    if (!words.length) { close(); return; }
    matches = state.movies
      .filter((m) => !state.picks.includes(m.id) && words.every((w) => m._search.includes(w)))
      .sort((a, b) => b.n_ratings - a.n_ratings).slice(0, 8).map((m) => m.id);
    hl = 0;
    draw();
  });
  input.addEventListener("keydown", (e) => {
    if (list.hidden || !matches.length) return;
    if (e.key === "ArrowDown") { hl = Math.min(hl + 1, matches.length - 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { hl = Math.max(hl - 1, 0); draw(); e.preventDefault(); }
    else if (e.key === "Enter") { choose(matches[hl]); e.preventDefault(); }
    else if (e.key === "Escape") close();
  });
  input.addEventListener("blur", close);
}

function setupControls(rerank) {
  state.lam = rerank.lambda;
  state.cap = rerank.cap.kmeans;
  $("lambda").value = String(state.lam);
  $("lambda-value").textContent = state.lam.toFixed(2);
  document.querySelectorAll("[data-cap]").forEach((b) => b.classList.toggle("active", Number(b.dataset.cap) === state.cap));

  $("lambda").addEventListener("input", (e) => {
    state.lam = Number(e.target.value);
    $("lambda-value").textContent = state.lam.toFixed(2);
    recommend();
  });
  document.querySelectorAll("[data-method]").forEach((b) => b.addEventListener("click", () => {
    state.method = b.dataset.method;
    document.querySelectorAll("[data-method]").forEach((x) => x.classList.toggle("active", x === b));
    $("control-lambda").hidden = state.method !== "mmr";
    $("control-cap").hidden = state.method !== "cap";
    recommend();
  }));
  document.querySelectorAll("[data-cap]").forEach((b) => b.addEventListener("click", () => {
    state.cap = Number(b.dataset.cap);
    document.querySelectorAll("[data-cap]").forEach((x) => x.classList.toggle("active", x === b));
    recommend();
  }));
  $("btn-example").addEventListener("click", () => {
    state.picks = EXAMPLE_PICKS
      .map((t) => state.movies.find((m) => m.title.startsWith(t)))
      .filter(Boolean).map((m) => m.id);
    update();
  });
  $("btn-clear").addEventListener("click", () => { state.picks = []; update(); });
}

// ------------------------------------------------------------------ charts

function chartDefaults() {
  Chart.defaults.font.family = '"Archivo", "Helvetica Neue", Arial, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = MUTED;
  Chart.defaults.animation = false;
  Chart.defaults.maintainAspectRatio = false;
  Object.assign(Chart.defaults.plugins.tooltip, {
    backgroundColor: "#fff", titleColor: INK, bodyColor: INK, borderColor: "#cdbca8", borderWidth: 1,
    padding: 10, cornerRadius: 0, boxPadding: 4, usePointStyle: true,
  });
}

function legend(ul, items) {
  ul.replaceChildren(...items.map(({ name, color, dot }) => {
    const li = el("li");
    const sw = el("span", { class: dot ? "sw dot" : "sw" });
    sw.style.background = color;
    li.append(sw, document.createTextNode(name));
    return li;
  }));
}

function lineChart(canvas, sim, metric) {
  const fmt = SIM_METRICS[metric].fmt;
  return new Chart(canvas, {
    type: "line",
    data: {
      labels: sim.rounds,
      datasets: POLICIES.map((p) => ({
        label: p.name, data: sim.policies[p.key][metric].mean,
        borderColor: p.color, backgroundColor: p.color, borderWidth: 2,
        pointRadius: 0, pointHoverRadius: 5, pointHoverBorderColor: "#fff", pointHoverBorderWidth: 2, tension: 0.15,
      })),
    },
    options: {
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: (it) => `Round ${it[0].label}`, label: (c) => ` ${c.dataset.label}: ${fmt(c.parsed.y)}` } },
      },
      scales: {
        x: { title: { display: true, text: "round" }, grid: { display: false }, ticks: { maxTicksLimit: 7 } },
        y: { grid: { color: GRID }, border: { display: false }, ticks: { callback: (v) => (fmt === pct ? pct(v, 0) : fmt(v)) } },
      },
    },
  });
}

function setupSimulation(sim) {
  legend($("sim-legend"), POLICIES);
  lineChart($("chart-ctr"), sim, "CTR");

  const select = $("sim-metric");
  for (const m of sim.metrics.filter((m) => m !== "CTR")) {
    select.append(el("option", { value: m }, SIM_METRICS[m].label));
  }
  select.value = "CumCoverage";
  let chart = lineChart($("chart-metric"), sim, "CumCoverage");
  select.addEventListener("change", () => {
    chart.destroy();
    chart = lineChart($("chart-metric"), sim, select.value);
  });

  const last = (p, m) => sim.policies[p][m].mean.at(-1);
  const lastStd = (p, m) => sim.policies[p][m].std.at(-1);
  const n = sim.rounds.length;
  $("sim-takeaway").textContent =
    `After ${n} rounds, plain SVD had shown ${pct(last("svd", "CumCoverage"))} of the catalogue and MMR ${pct(last("mmr", "CumCoverage"))}. ` +
    `Their final click rates were ${pct(last("svd", "CTR"))} and ${pct(last("mmr", "CTR"))}, a gap smaller than the spread between seeds. ` +
    `Click rate falls for every policy because users click their best matches early and are never shown them again; ` +
    `exposure stayed stuck rather than collapsing, and more variety did not raise Discovery.`;

  const metrics = sim.metrics;
  const table = $("sim-table");
  const head = el("tr");
  head.append(el("th", {}, `Policy (round ${n})`), ...metrics.map((m) => el("th", {}, m)));
  table.replaceChildren(el("thead"), el("tbody"));
  table.tHead.append(head);
  for (const p of POLICIES) {
    const tr = el("tr");
    tr.append(el("td", {}, p.name));
    for (const m of metrics) {
      const td = el("td", {}, SIM_METRICS[m].fmt(last(p.key, m)));
      td.append(el("span", { class: "pm" }, "± " + SIM_METRICS[m].fmt(lastStd(p.key, m))));
      tr.append(td);
    }
    table.tBodies[0].append(tr);
  }
}

function setupModels(baselines) {
  const cols = [
    ["RMSE", (v) => v.toFixed(3), "min"], ["NDCG@10", (v) => v.toFixed(3), "max"],
    ["Precision@10", (v) => v.toFixed(3), null], ["Coverage", (v) => pct(v), null],
    ["AvgPop", (v) => v.toFixed(0), null],
  ];
  const names = Object.keys(baselines.models);
  const contenders = names.filter((n) => !n.includes("reference"));
  const bestOf = (metric, dir) => contenders.reduce((a, b) =>
    (dir === "min" ? baselines.models[b][metric].mean < baselines.models[a][metric].mean
                   : baselines.models[b][metric].mean > baselines.models[a][metric].mean) ? b : a);
  const best = Object.fromEntries(cols.filter((c) => c[2]).map(([m, , d]) => [m, bestOf(m, d)]));

  const table = $("models-table");
  table.replaceChildren(el("thead"), el("tbody"));
  const head = el("tr");
  head.append(el("th", {}, "Model"), ...cols.map(([m]) => el("th", {}, m === "AvgPop" ? "Avg popularity" : m)));
  table.tHead.append(head);
  for (const n of names) {
    const tr = el("tr");
    tr.append(el("td", {}, n));
    for (const [m, fmt] of cols) {
      const v = baselines.models[n][m];
      const td = el("td", { class: best[m] === n ? "best" : "" }, fmt(v.mean));
      if (v.std !== null) td.append(el("span", { class: "pm" }, "± " + fmt(v.std)));
      tr.append(td);
    }
    table.tBodies[0].append(tr);
  }
  $("selection-note").textContent =
    `Underlined: best RMSE and best NDCG@10 among the four algorithms. The selection rule fixed in advance, ` +
    `highest validation NDCG@10, picked ${baselines.best_model}; on test, ${best["NDCG@10"]} makes the best lists. ` +
    `Both are reported. The re-ranking and simulation build on SVD because they need personalised scores and film embeddings.`;

  const knn = baselines.models["ItemKNN"];
  if (knn) $("stat-knn").textContent = `${knn.RMSE.mean.toFixed(2)} / ${knn["NDCG@10"].mean.toFixed(3)}`;
}

function setupTradeoff(rerank) {
  const curve = rerank.curve;
  const pts = (method) => curve.filter((r) => r.method === method)
    .map((r) => ({ x: r.Coverage, y: r["NDCG@10"], param: r.param }));
  const mmrPts = pts("MMR").sort((a, b) => b.param - a.param);
  const sets = [
    { name: "MMR (λ from 1.0 down to 0.3)", color: "#CF9800", data: mmrPts, line: true, tip: (p) => `MMR λ = ${p.param.toFixed(1)}${p.param === 1 ? " (plain SVD)" : ""}` },
    { name: "K-Means cap (1, 2, 3)", color: "#2E78B8", data: pts("KMeans-cap"), tip: (p) => `K-Means cap ${p.param}` },
    { name: "DBSCAN cap (1, 2, 3)", color: "#B24A6E", data: pts("DBSCAN-cap"), tip: (p) => `DBSCAN cap ${p.param}` },
    { name: "Popularity (reference)", color: "#8F4E22", data: pts("Popularity (reference)"), tip: () => "Popularity" },
  ];
  legend($("tradeoff-legend"), sets.map((s) => ({ ...s, dot: true })));

  // selective direct labels: the two ends of the MMR dial
  const labelPlugin = {
    id: "endLabels",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.font = '600 12px "Archivo", sans-serif';
      ctx.fillStyle = INK;
      [[0, "λ = 1 (plain SVD)", 10, -8, "left"], [meta.data.length - 1, "λ = 0.3", -10, 4, "right"]].forEach(([i, text, dx, dy, align]) => {
        const p = meta.data[i];
        ctx.textAlign = align;
        if (p) ctx.fillText(text, p.x + dx, p.y + dy);
      });
      ctx.restore();
    },
  };

  new Chart($("chart-tradeoff"), {
    type: "scatter",
    data: {
      datasets: sets.map((s) => ({
        label: s.name, data: s.data, showLine: !!s.line,
        borderColor: s.color, backgroundColor: s.color, borderWidth: s.line ? 2 : 0,
        pointRadius: 5, pointHoverRadius: 7, pointBorderColor: "#fff", pointBorderWidth: 2,
        pointStyle: s.name.startsWith("Popularity") ? "rect" : "circle",
      })),
    },
    options: {
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (it) => sets[it[0].datasetIndex].tip(it[0].raw),
            label: (c) => ` NDCG@10 ${c.parsed.y.toFixed(3)} · coverage ${pct(c.parsed.x)}`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "catalogue coverage (share of films recommended to anyone)" }, grid: { color: GRID }, border: { display: false }, ticks: { callback: (v) => pct(v, 0) }, min: 0 },
        y: { title: { display: true, text: "NDCG@10" }, grid: { color: GRID }, border: { display: false } },
      },
    },
    plugins: [labelPlugin],
  });
}

// ------------------------------------------------------------------ start

async function main() {
  const [movies, factors, clusters, rerank, baselines, sim] = await Promise.all(
    ["movies.json", "item_factors.json", "clusters.json", "rerank.json", "baselines.json", "simulation.json"].map(loadJSON));

  state.movies = movies.movies.map((m) => ({ ...m, _search: normalise(m.title) }));
  state.k = factors.k;
  state.Q = Float64Array.from(factors.item_vectors.flat());
  state.Qn = new Float64Array(state.Q.length);
  for (let i = 0; i < state.movies.length; i++) {
    let norm = 0;
    for (let a = 0; a < state.k; a++) norm += state.Q[i * state.k + a] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let a = 0; a < state.k; a++) state.Qn[i * state.k + a] = state.Q[i * state.k + a] / norm;
  }
  state.kmeans = clusters.kmeans;

  const last = (p, m) => sim.policies[p][m].mean.at(-1);
  $("stat-coverage").textContent = `${pct(last("svd", "CumCoverage"))} → ${pct(last("mmr", "CumCoverage"))}`;
  $("stat-ctr").textContent = `${pct(last("svd", "CTR"))} vs ${pct(last("mmr", "CTR"))}`;

  setupSearch();
  setupControls(rerank);
  update();

  if (window.Chart) {
    chartDefaults();
    setupSimulation(sim);
    setupModels(baselines);
    setupTradeoff(rerank);
  } else {
    setupModels(baselines);
    document.querySelectorAll(".chart-box").forEach((b) => { b.textContent = "Charts could not load (Chart.js unavailable)."; });
  }
}

main().catch((err) => {
  console.error(err);
  $("empty-state").textContent = `Could not load the data files (${err.message}). If you opened index.html directly, serve the folder over HTTP instead.`;
});
