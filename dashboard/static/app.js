"use strict";
/* Peephole Parley dashboard. Plain JavaScript, no libraries.
   Text that comes from the server or the user is always inserted with textContent. */

const RULE_NAMES = {
  CF: "Constant folding", AS: "Algebraic simplification", SR: "Strength reduction", CP: "Copy propagation",
  RA: "Reassociation", CSE: "Common sub-expression elimination", RS: "Redundant statements", DCE: "Dead-code elimination",
};
const SERIES = [
  { key: "ml", label: "Model's pick", color: "var(--s1)", shape: "circle" },
  { key: "best_fixed", label: "Best fixed pipeline", color: "var(--s2)", shape: "square" },
  { key: "oracle", label: "Oracle (best possible)", color: "var(--s3)", shape: "diamond" },
  { key: "textbook", label: "Textbook order", color: "var(--s4)", shape: "triangle" },
];
const STRAT_COLOR = { ml: "var(--s1)", fixed: "var(--s2)", oracle: "var(--s3)", textbook: "var(--s4)", fixpoint: "var(--sx)" };
const MODEL_NAMES = {
  DecisionTree: "Decision tree", RandomForest: "Random forest", HistGradBoost: "Gradient boosting",
  LogisticReg: "Logistic regression", kNN: "k-nearest neighbours", Ridge: "Ridge regression",
};
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (id) => document.getElementById(id);
const state = { budget: 4, summary: [], budgets: [], info: null, last: null, shown: "ml", examples: [], seed: 7 };
const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, text) {
  const n = document.createElementNS(tag === "svg" || SVG_TAGS.has(tag) ? NS : "http://www.w3.org/1999/xhtml", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
const SVG_TAGS = new Set(["g", "path", "line", "rect", "circle", "text", "polygon"]);
const fmt = (x, d = 2) => Number(x).toFixed(d);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
};

/* ---------------- theme ---------------- */
function initTheme() {
  const saved = (() => { try { return localStorage.getItem("parley-theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme").addEventListener("click", () => {
    const dark = getComputedStyle(document.documentElement).colorScheme === "dark";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("parley-theme", next); } catch { /* storage may be blocked */ }
    drawCharts();
  });
}

/* ---------------- budget filter ---------------- */
function buildBudget() {
  const seg = $("budget");
  seg.textContent = "";
  for (const b of state.budgets) {
    const btn = el("button", { type: "button", role: "radio", "aria-checked": String(b === state.budget) }, String(b));
    btn.addEventListener("click", () => setBudget(b));
    btn.addEventListener("keydown", (e) => {
      const i = state.budgets.indexOf(state.budget);
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) { e.preventDefault(); setBudget(state.budgets[Math.max(0, Math.min(state.budgets.length - 1, i + d))]); seg.querySelector('[aria-checked="true"]').focus(); }
    });
    seg.append(btn);
  }
  $("b-hint").textContent = hint(state.budget);
}
function hint(b) {
  return b === 8 ? "All eight rules run, so only their order matters." : `Each pipeline may use ${b} of the 8 rules, in any order.`;
}
async function setBudget(b) {
  state.budget = b;
  for (const btn of $("budget").children) btn.setAttribute("aria-checked", String(btn.textContent === String(b)));
  $("b-hint").textContent = hint(b);
  drawCharts();
  await loadBudgetInfo();
  if (state.last) run();
}

/* ---------------- workbench ---------------- */
function setError(msg) { const e = $("error"); e.hidden = !msg; e.textContent = msg || ""; }

async function run() {
  const btn = $("run");
  btn.disabled = true; $("status").textContent = "Optimizing…"; setError("");
  try {
    const res = await api("/api/optimize", { method: "POST", body: JSON.stringify({ code: $("code").value, budget: state.budget }) });
    state.last = res; state.shown = "ml";
    renderStrategies(); renderOutput(); await renderStrip(res.results.ml, res.base_cost);
    $("status").textContent = "";
  } catch (e) {
    setError(e.message); $("status").textContent = "";
  } finally { btn.disabled = false; }
}

async function renderStrip(r, base) {
  const strip = $("strip");
  strip.textContent = "";
  const chips = [];
  if (r.repeated) {
    const c = chip("All 8", `${r.effort} rule runs`, null); strip.append(c); chips.push([c, null]);
  } else {
    r.steps.forEach((s, i) => {
      if (i) strip.append(el("div", { class: "link" }));
      const delta = s.after - s.before;
      const c = chip(s.rule, delta ? `${delta}` : "no change", delta < 0);
      c.title = RULE_NAMES[s.rule];
      strip.append(c); chips.push([c, s]);
    });
  }
  const total = el("div", { class: "total" });
  const big = el("div", { class: "big" });
  const num = el("b", {}, String(base));
  big.append(num, el("span", {}, " → "), el("b", { id: "endcost" }, String(base)));
  const pct = base ? Math.round((100 * (base - r.cost)) / base) : 0;
  const note = el("div", { class: "note" });
  note.append(document.createTextNode("static cost, "), el("b", {}, `${pct}% lower`));
  total.append(big, note);
  strip.append(total);
  const end = total.querySelector("#endcost");
  if (reduceMotion) { chips.forEach(([c]) => { c.className += " lit"; }); end.textContent = String(r.cost); return; }
  chips.forEach(([c]) => c.classList.add("dim"));
  let cur = base;
  for (const [c, s] of chips) {
    await new Promise((ok) => setTimeout(ok, 170));
    c.classList.remove("dim"); c.classList.add("lit");
    cur = s ? s.after : r.cost; end.textContent = String(cur);
  }
}
function chip(rule, note, gain) {
  const c = el("div", { class: "chip" + (gain ? " gain" : "") });
  c.append(el("b", {}, rule), el("small", {}, note));
  return c;
}

function renderOutput() {
  const r = state.last.results[state.shown];
  $("out-title").textContent = `Optimized · ${r.label} · cost ${r.cost}`;
  const before = new Set($("code").value.split("\n").map((l) => l.trim()));
  const pre = $("out"); pre.textContent = "";
  const lines = r.output.replace(/\n$/, "").split("\n");
  for (const l of lines) {
    const d = el("span", { class: "ln" + (before.has(l.trim()) || !l.trim() ? "" : " new") }, l || " ");
    pre.append(d);
  }
}

function renderStrategies() {
  const wrap = $("strategies"); wrap.textContent = "";
  const { results, base_cost } = state.last;
  for (const [key, r] of Object.entries(results)) {
    const row = el("button", { class: "srow", type: "button", "aria-pressed": String(key === state.shown) });
    const nm = el("div", { class: "nm" }, r.label);
    const pipe = el("div", { class: "pipe" });
    r.pipeline.forEach((p) => pipe.append(el("span", {}, p)));
    const sub = el("small", {}, r.repeated ? `${r.effort} rule runs, repeated until stable` : `${r.effort} rule runs`);
    nm.append(sub);
    const mid = el("div");
    const bar = el("div", { class: "bar" }); bar.style.setProperty("--c", STRAT_COLOR[key]);
    const fill = el("i"); fill.style.width = `${Math.max(2, (100 * r.cost) / base_cost)}%`; bar.append(fill);
    mid.append(bar, pipe);
    const val = el("div", { class: "val" }, String(r.cost));
    const ok = el("small", { class: r.same_behaviour ? "ok" : "nope" }, r.same_behaviour ? "Same behaviour as the original" : "Behaviour changed");
    val.append(ok);
    row.append(nm, mid, val);
    row.addEventListener("click", async () => {
      state.shown = key; renderStrategies(); renderOutput(); await renderStrip(r, base_cost);
    });
    wrap.append(row);
  }
}

async function loadExamples() {
  state.examples = await api("/api/examples");
  const sel = $("example"); sel.textContent = "";
  state.examples.forEach((e, i) => sel.append(el("option", { value: String(i) }, e.name.replace(/_/g, " "))));
  sel.addEventListener("change", () => { $("code").value = state.examples[+sel.value].code; run(); });
  if (state.examples.length) $("code").value = state.examples[0].code;
}

/* ---------------- charts ---------------- */
const tip = $("tip");
function showTip(html, x, y) {
  tip.textContent = ""; tip.append(html); tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${Math.min(x + 16, innerWidth - w - 8)}px`;
  tip.style.top = `${Math.min(Math.max(8, y - h / 2), innerHeight - h - 8)}px`;
}
const hideTip = () => { tip.hidden = true; };

function marker(shape, cx, cy, color, r = 5) {
  const common = { fill: color, stroke: "var(--surface)", "stroke-width": 2 };
  if (shape === "square") return el("rect", { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, rx: 1.5, ...common });
  if (shape === "diamond") return el("polygon", { points: `${cx},${cy - r - 1.5} ${cx + r + 1.5},${cy} ${cx},${cy + r + 1.5} ${cx - r - 1.5},${cy}`, ...common });
  if (shape === "triangle") return el("polygon", { points: `${cx},${cy - r - 1} ${cx + r + 1},${cy + r} ${cx - r - 1},${cy + r}`, ...common });
  return el("circle", { cx, cy, r, ...common });
}

function drawCharts() {
  if (!state.summary.length) return;
  drawLegend(); drawCost(); drawSave(); drawTable();
}

function drawLegend() {
  const lg = $("legendCost"); lg.textContent = "";
  for (const s of SERIES) {
    const sp = el("span");
    const svg = el("svg", { viewBox: "0 0 22 12" });
    svg.append(el("line", { x1: 0, x2: 22, y1: 6, y2: 6, stroke: s.color, "stroke-width": 2 }), marker(s.shape, 11, 6, s.color, 3.5));
    sp.append(svg, document.createTextNode(s.label)); lg.append(sp);
  }
  const sp = el("span"); const svg = el("svg", { viewBox: "0 0 22 12" });
  svg.append(el("line", { x1: 0, x2: 22, y1: 6, y2: 6, stroke: "var(--ink3)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
  sp.append(svg, document.createTextNode("No optimization")); lg.append(sp);
}

function drawCost() {
  const host = $("costPlot"); host.textContent = "";
  const W = Math.max(320, host.clientWidth), H = W < 520 ? 280 : 340;
  const m = { l: 42, r: W < 520 ? 12 : 126, t: 14, b: 40 };
  const D = state.summary, xs = D.map((d) => d.budget);
  const x = (b) => m.l + ((b - 1) / (xs.length - 1)) * (W - m.l - m.r);
  const lo = 40, hi = 112, y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Average cost by number of rules allowed, for four strategies. The table view has every value.", tabindex: "0" });
  const sel = D.findIndex((d) => d.budget === state.budget);
  svg.append(el("rect", { class: "sel", x: x(state.budget) - 18, y: m.t, width: 36, height: H - m.t - m.b, rx: 6 }));
  for (const t of [40, 60, 80, 100]) {
    svg.append(el("line", { class: "grid", x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }), el("text", { class: "num", x: m.l - 8, y: y(t) + 4, "text-anchor": "end" }, String(t)));
  }
  svg.append(el("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  xs.forEach((b) => svg.append(el("text", { class: "num", x: x(b), y: H - m.b + 18, "text-anchor": "middle" }, String(b))));
  svg.append(el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, "Rules allowed per program"));
  const base = D[0].mean_cost.none;
  svg.append(el("line", { x1: m.l, x2: W - m.r, y1: y(base), y2: y(base), stroke: "var(--ink3)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
  const ends = [];
  for (const s of SERIES) {
    const pts = D.map((d) => [x(d.budget), y(d.mean_cost[s.key])]);
    svg.append(el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(""), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }));
    pts.forEach((p) => svg.append(marker(s.shape, p[0], p[1], s.color, 4)));
    ends.push({ s, y: pts[pts.length - 1][1] });
  }
  if (W >= 520) {  // direct labels with a minimum spacing so they never collide
    ends.sort((a, b) => a.y - b.y);
    let prev = -99;
    for (const e of ends) { e.ly = Math.max(e.y, prev + 14); prev = e.ly; }
    for (const e of ends) {
      svg.append(el("line", { x1: x(8) + 8, x2: x(8) + 14, y1: e.y, y2: e.ly, stroke: "var(--ink3)", "stroke-width": 1 }),
        el("text", { x: x(8) + 18, y: e.ly + 4, style: "fill:var(--ink)" }, e.s.label.replace(" (best possible)", "")));
    }
    svg.append(el("text", { x: W - m.r + 6, y: y(base) - 6 }, `No optimization ${fmt(base, 1)}`));
  }
  const cross = el("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: "var(--ink)", "stroke-width": 1, visibility: "hidden" });
  svg.append(cross);
  const show = (i, ev) => {
    const d = D[i]; cross.setAttribute("x1", x(d.budget)); cross.setAttribute("x2", x(d.budget)); cross.setAttribute("visibility", "visible");
    const box = document.createElement("div");
    const h = document.createElement("h4"); h.textContent = `${d.budget} ${d.budget === 1 ? "rule" : "rules"} allowed`; box.append(h);
    [...SERIES].sort((a, b) => d.mean_cost[a.key] - d.mean_cost[b.key]).forEach((s) => {
      const r = document.createElement("div"); r.className = "r"; r.style.setProperty("--c", s.color);
      const k = document.createElement("span"); k.className = "k";
      const n = document.createElement("span"); n.textContent = s.label;
      const v = document.createElement("b"); v.textContent = fmt(d.mean_cost[s.key]);
      r.append(k, n, v); box.append(r);
    });
    const rect = svg.getBoundingClientRect();
    showTip(box, ev ? ev.clientX : rect.left + (x(d.budget) / W) * rect.width, ev ? ev.clientY : rect.top + 80);
  };
  const near = (ev) => { const r = svg.getBoundingClientRect(); const px = ((ev.clientX - r.left) / r.width) * W; return xs.reduce((bi, b, i) => (Math.abs(x(b) - px) < Math.abs(x(xs[bi]) - px) ? i : bi), 0); };
  svg.addEventListener("pointermove", (ev) => show(near(ev), ev));
  svg.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  svg.addEventListener("click", (ev) => setBudget(xs[near(ev)]));
  let k = sel;
  svg.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") { ev.preventDefault(); k = Math.max(0, Math.min(xs.length - 1, k + (ev.key === "ArrowRight" ? 1 : -1))); show(k); }
  });
  svg.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  host.append(svg);
}

function drawSave() {
  const host = $("savePlot"); host.textContent = "";
  const D = state.summary, rowH = 34, W = Math.max(300, host.clientWidth);
  const m = { l: 40, r: 74, t: 8, b: 40 }, H = m.t + D.length * rowH + m.b;
  const maxX = 7.5, x = (v) => m.l + (v / maxX) * (W - m.l - m.r);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Average saving of the model over the best fixed pipeline, with 95 percent range, for each budget." });
  for (const t of [0, 2, 4, 6]) {
    svg.append(el("line", { class: "grid", x1: x(t), x2: x(t), y1: m.t, y2: H - m.b }), el("text", { class: "num", x: x(t), y: H - m.b + 16, "text-anchor": "middle" }, String(t)));
  }
  svg.append(el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, "Cost units saved per program"));
  svg.append(el("text", { x: W - 2, y: m.t + 2, "text-anchor": "end", style: "font-size:11px" }, "Gap closed"));
  D.forEach((d, i) => {
    const cy = m.t + 14 + i * rowH, [lo, hi] = d.ml_vs_fixed.ci95, v = d.ml_vs_fixed.mean_gain, on = d.budget === state.budget;
    if (on) svg.append(el("rect", { class: "sel", x: 0, y: cy - rowH / 2, width: W, height: rowH, rx: 6 }));
    svg.append(el("text", { class: "num", x: m.l - 14, y: cy + 4, "text-anchor": "end", style: on ? "fill:var(--ink);font-weight:600" : "" }, String(d.budget)));
    svg.append(el("line", { x1: x(lo), x2: x(hi), y1: cy, y2: cy, stroke: "var(--s1)", "stroke-width": 2 }),
      el("line", { x1: x(lo), x2: x(lo), y1: cy - 5, y2: cy + 5, stroke: "var(--s1)", "stroke-width": 2 }),
      el("line", { x1: x(hi), x2: x(hi), y1: cy - 5, y2: cy + 5, stroke: "var(--s1)", "stroke-width": 2 }),
      marker("circle", x(v), cy, "var(--s1)", 5));
    svg.append(el("text", { class: "num", x: W - 2, y: cy + 4, "text-anchor": "end", style: "fill:var(--ink)" }, `${Math.round(d.headroom_recovered_pct)}%`));
    const hit = el("rect", { class: "hit", x: 0, y: cy - rowH / 2, width: W, height: rowH, tabindex: "0", role: "button", "aria-label": `Budget ${d.budget}: saves ${fmt(v)}, range ${fmt(lo)} to ${fmt(hi)}` });
    const open = (ev) => {
      const box = document.createElement("div"), h = document.createElement("h4"); h.textContent = `${d.budget} ${d.budget === 1 ? "rule" : "rules"} allowed`; box.append(h);
      for (const [a, b] of [["Saves", `${fmt(v)}`], ["95% range", `${fmt(lo)} to ${fmt(hi)}`], ["Gap to oracle closed", `${Math.round(d.headroom_recovered_pct)}%`], ["Better / same / worse", `${d.ml_vs_fixed.better} / ${d.ml_vs_fixed.same} / ${d.ml_vs_fixed.worse}`]]) {
        const r = document.createElement("div"); r.className = "r"; const n = document.createElement("em"); n.textContent = a; const bb = document.createElement("b"); bb.textContent = b; r.append(n, bb); box.append(r);
      }
      const rc = hit.getBoundingClientRect(); showTip(box, ev && ev.clientX ? ev.clientX : rc.left + 60, ev && ev.clientY ? ev.clientY : rc.top + 10);
    };
    hit.addEventListener("pointermove", open); hit.addEventListener("pointerleave", hideTip); hit.addEventListener("focus", () => open());
    hit.addEventListener("blur", hideTip); hit.addEventListener("click", () => setBudget(d.budget));
    hit.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setBudget(d.budget); } });
    svg.append(hit);
  });
  host.append(svg);
}

function drawTable() {
  const host = $("tableView"); host.textContent = "";
  const cols = ["Budget", "No optimization", "Textbook", "Best fixed", "Model", "Oracle", "Model saves (95% range)", "Gap closed", "Behaviour failures"];
  const t = document.createElement("table"), thead = t.createTHead().insertRow();
  cols.forEach((c) => { const th = document.createElement("th"); th.scope = "col"; th.textContent = c; thead.append(th); });
  const body = t.createTBody();
  for (const d of state.summary) {
    const r = body.insertRow(); if (d.budget === state.budget) r.className = "on";
    const m = d.mean_cost;
    [d.budget, fmt(m.none, 1), fmt(m.textbook, 1), fmt(m.best_fixed), fmt(m.ml), fmt(m.oracle),
      `+${fmt(d.ml_vs_fixed.mean_gain)} [${fmt(d.ml_vs_fixed.ci95[0])}, ${fmt(d.ml_vs_fixed.ci95[1])}]`,
      `${Math.round(d.headroom_recovered_pct)}%`, `${d.equivalence_failures} of ${d.equivalence_checked}`].forEach((v) => { r.insertCell().textContent = String(v); });
  }
  host.append(t);
}

/* ---------------- models, portfolio, trust ---------------- */
function modelLabel(k) {
  if (k === "Best fixed pipeline") return ["Best fixed pipeline", "One pipeline for every program"];
  const [base, kind] = k.replace(")", "").split(" (");
  const nm = MODEL_NAMES[base] || base;
  return [base === "Ridge" ? nm : `${nm} (${kind === "reg" ? "regressor" : "classifier"})`, kind === "reg" ? "Predicts the regret of each pipeline" : "Predicts the best pipeline"];
}
function drawModels() {
  const info = state.info; if (!info) return;
  const host = $("models"); host.textContent = "";
  const entries = Object.entries(info.cv).sort((a, b) => a[1].mean_regret - b[1].mean_regret);
  const max = Math.max(...entries.map(([, v]) => v.mean_regret));
  $("mod-sub").textContent = `Nine models for ${state.budget} ${state.budget === 1 ? "rule" : "rules"}, cross-validated on the 1,600 training programs. Bars show regret: the extra cost compared with the best pipeline for each program (shorter is better).`;
  for (const [k, v] of entries) {
    const [nm, how] = modelLabel(k);
    const row = el("div", { class: "mrow" + (k === info.winner ? " win" : "") + (k === "Best fixed pipeline" ? " fixed" : "") });
    const name = el("div", { class: "nm" }, nm);
    if (k === info.winner) name.append(el("span", { class: "badge" }, "Winner"));
    const sub = el("small", {}, k === "Best fixed pipeline" ? how : `${how} · picks the best one ${Math.round(v.optimal_rate * 100)}% of the time`);
    name.append(sub);
    const tr = el("div", { class: "track" }); const bar = el("i"); bar.style.width = `${Math.max(1.5, (100 * v.mean_regret) / max)}%`; tr.append(bar);
    row.append(name, tr, el("div", { class: "v" }, fmt(v.mean_regret)));
    host.append(row);
  }
}
function drawPortfolio() {
  const info = state.info; if (!info) return;
  const same = (a, b) => a.join() === b.join();
  $("port-sub").textContent = `${info.portfolio.length} pipelines of ${state.budget} ${state.budget === 1 ? "rule" : "rules"}. The model picks one per program. The first is the best single pipeline overall.`;
  const ol = $("portfolio"); ol.textContent = "";
  info.portfolio.forEach((p, i) => {
    const li = el("li", { class: i === 0 ? "first" : "" });
    li.append(el("span", { class: "idx" }, String(i)));
    const pipe = el("div", { class: "pipe" }); p.forEach((r) => { const s = el("span", {}, r); s.title = RULE_NAMES[r]; pipe.append(s); });
    li.append(pipe);
    const tags = [i === 0 ? "best fixed" : "", same(p, info.textbook) ? "textbook order" : ""].filter(Boolean).join(", ");
    if (tags) li.append(el("span", { class: "tag" }, tags));
    ol.append(li);
  });
  const key = $("rulekey");
  if (!key.children.length) for (const [k, v] of Object.entries(RULE_NAMES)) { const d = el("div"); d.append(el("dt", {}, k), el("dd", {}, v)); key.append(d); }
}
function drawFacts() {
  const total = state.summary.reduce((a, d) => a + d.equivalence_checked, 0);
  const bad = state.summary.reduce((a, d) => a + d.equivalence_failures, 0);
  const facts = [[`${bad} of ${total.toLocaleString()}`, "optimized programs changed behaviour"], ["22", "automated tests, all passing"],
    ["1,450", "programs fuzz-test each rule against the original"], ["64 of 1,200", "programs exposed a bug we planted on purpose"]];
  const host = $("facts"); host.textContent = "";
  for (const [b, s] of facts) { const f = el("div", { class: "fact" }); f.append(el("b", {}, b), el("span", {}, s)); host.append(f); }
}
async function loadBudgetInfo() {
  state.info = await api(`/api/budget?b=${state.budget}`);
  drawModels(); drawPortfolio();
}

/* ---------------- start ---------------- */
async function init() {
  initTheme();
  try {
    const s = await api("/api/summary");
    state.summary = s.summary; state.budgets = s.budgets;
    buildBudget(); drawFacts(); drawCharts();
    await Promise.all([loadExamples(), loadBudgetInfo()]);
    $("run").addEventListener("click", run);
    $("random").addEventListener("click", async () => { const r = await api(`/api/random?seed=${state.seed++}`); $("code").value = r.code; $("example").selectedIndex = -1; run(); });
    $("tableToggle").addEventListener("click", (e) => {
      const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
      e.currentTarget.setAttribute("aria-pressed", String(on)); e.currentTarget.textContent = on ? "View as charts" : "View as table";
      $("charts").hidden = on; $("tableView").hidden = !on; if (!on) drawCharts();
    });
    $("code").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); });
    let t; new ResizeObserver(() => { clearTimeout(t); t = setTimeout(() => { if (!$("charts").hidden) drawCharts(); }, 120); }).observe(document.body);
    run();
  } catch (e) { setError(`The dashboard cannot reach its server: ${e.message}`); }
}
init();
