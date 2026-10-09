// Simulation competitions: pick a four-seat roster and run seat-cycled matches in a worker.

import { AGENTS } from "./agents.js";
import { $, esc, fmt, pct } from "./ui.js";

const ORDER = ["lookahead", "improved", "heuristic", "simple", "random"];
const PRESETS = [
  { name: "Lookahead vs heuristics", roster: ["lookahead", "improved", "heuristic", "random"] },
  { name: "Heuristic ladder", roster: ["improved", "heuristic", "simple", "random"] },
  { name: "Two lookaheads", roster: ["lookahead", "lookahead", "improved", "improved"] },
];

let worker = null;

export function renderArena(view) {
  if (worker) { worker.terminate(); worker = null; }
  const opt = (sel) => ORDER.map((k) => `<option value="${k}" ${k === sel ? "selected" : ""}>${AGENTS[k].name}</option>`).join("");
  const start = PRESETS[0].roster;

  view.innerHTML = `
    <div class="eyebrow">Arena</div>
    <h1>Simulation competitions</h1>
    <p class="lede">Pick four agents and run a tournament in your browser. With seat cycling on, every distinct seating is played the same number of times, which is how the repo's own benchmarks remove seat-order bias.</p>

    <div class="panel">
      <div class="btn-row" style="margin-bottom:0.9rem">${PRESETS.map((p, i) => `<button class="btn" data-preset="${i}">${esc(p.name)}</button>`).join("")}</div>
      <div class="form-row">
        ${[0, 1, 2, 3].map((i) => `<label class="field">Seat ${i + 1}<select data-seat="${i}">${opt(start[i])}</select></label>`).join("")}
        <label class="field">Matches per seating<input id="gps" type="number" min="1" max="2000" value="25" style="width:7rem"></label>
        <label class="field">Holes<select id="holes"><option>1</option><option>3</option><option selected>9</option></select></label>
        <label class="field" style="grid-auto-flow:column;align-items:center;gap:0.4rem"><input type="checkbox" id="cycle" checked> Seat cycling</label>
      </div>
      <div class="btn-row" style="margin-top:0.9rem;align-items:center">
        <button class="btn primary" id="run">Run competition</button>
        <button class="btn" id="stop" disabled>Stop</button>
        <span class="muted small" id="plan"></span>
      </div>
      <div class="progress" style="margin-top:0.9rem"><span id="bar"></span></div>
    </div>

    <div id="results" style="margin-top:1.4rem"></div>

    <h2>Agents</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Agent</th><th>Code</th><th>How it plays</th></tr></thead>
      <tbody>${ORDER.map((k) => `<tr><td><b>${AGENTS[k].name}</b></td><td class="mono">${AGENTS[k].code}</td><td style="white-space:normal">${esc(AGENTS[k].blurb)}</td></tr>`).join("")}</tbody>
    </table></div>
    <p class="muted small">The DQN champion and LLM players run in Python and aren't available in the browser yet. Funding the research agenda brings them here.</p>`;

  const seats = () => [...view.querySelectorAll("[data-seat]")].map((s) => s.value);
  const updatePlan = () => {
    const r = seats();
    const perms = $("#cycle", view).checked ? distinctPerms(r) : 1;
    const n = perms * Math.max(1, +$("#gps", view).value || 1);
    $("#plan", view).textContent = `${perms} seating${perms > 1 ? "s" : ""} × ${$("#gps", view).value} = ${n.toLocaleString()} matches`;
  };
  const form = $(".panel", view);
  form.addEventListener("change", updatePlan);
  form.addEventListener("input", updatePlan);
  updatePlan();

  view.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    const r = PRESETS[+b.dataset.preset].roster;
    view.querySelectorAll("[data-seat]").forEach((s, i) => { s.value = r[i]; });
    updatePlan();
  }));

  $("#run", view).addEventListener("click", () => run(view, seats()));
  $("#stop", view).addEventListener("click", () => {
    if (worker) { worker.terminate(); worker = null; }
    $("#run", view).disabled = false; $("#stop", view).disabled = true;
    $("#plan", view).textContent = "Stopped.";
  });
}

function distinctPerms(r) {
  const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
  const counts = {};
  r.forEach((k) => { counts[k] = (counts[k] || 0) + 1; });
  return Object.values(counts).reduce((d, c) => d / fact(c), fact(r.length));
}

function run(view, roster) {
  if (worker) worker.terminate();
  const gps = Math.min(2000, Math.max(1, +$("#gps", view).value || 1));
  const holes = +$("#holes", view).value;
  const cycle = $("#cycle", view).checked;
  const seed = (Math.random() * 2 ** 32) >>> 0;
  $("#run", view).disabled = true; $("#stop", view).disabled = false;
  $("#bar", view).style.width = "0%";

  worker = new Worker(new URL("./arena-worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (ev) => {
    const d = ev.data;
    if (d.type === "progress") {
      $("#bar", view).style.width = `${(100 * d.done) / d.total}%`;
      return;
    }
    $("#bar", view).style.width = "100%";
    $("#run", view).disabled = false; $("#stop", view).disabled = true;
    worker.terminate(); worker = null;
    showResults(view, roster, d, { gps, holes, cycle, seed });
  };
  worker.onerror = (e) => {
    $("#plan", view).textContent = `Simulation failed: ${e.message}`;
    $("#run", view).disabled = false; $("#stop", view).disabled = true;
  };
  worker.postMessage({ roster, gamesPerSeating: gps, holes, cycle, seed });
}

function showResults(view, roster, d, cfg) {
  const best = d.labels[0].mean;
  const maxMean = Math.max(...d.labels.map((l) => l.mean));
  const rows = d.labels.map((l, i) => `
    <tr>
      <td class="rank">${i + 1}</td>
      <td><b>${AGENTS[l.key].name}</b></td>
      <td class="num">${fmt(l.mean)} <span class="muted">± ${fmt(l.ci)}</span></td>
      <td><div class="bar"><span style="width:${(100 * l.mean) / maxMean}%"></span></div></td>
      <td class="num">${pct(l.win)}</td>
      <td class="num">${l.mean === best ? "" : "+" + fmt(l.mean - best)}</td>
    </tr>`).join("");

  const md = [
    `Arena: ${roster.map((k) => AGENTS[k].code).join(",")} · ${d.seatings} seatings × ${cfg.gps} matches × ${cfg.holes} holes · seed ${cfg.seed}`,
    "",
    "| Agent | Avg score/hole | 95% CI | Win rate |",
    "|---|---|---|---|",
    ...d.labels.map((l) => `| ${AGENTS[l.key].name} | ${fmt(l.mean)} | ±${fmt(l.ci)} | ${pct(l.win)} |`),
  ].join("\n");

  $("#results", view).innerHTML = `
    <h2 style="margin-top:0">Results</h2>
    <p class="muted small">${d.matches.toLocaleString()} matches · ${d.seatings} seating${d.seatings > 1 ? "s" : ""} · ${cfg.holes} hole${cfg.holes > 1 ? "s" : ""} · ${(d.ms / 1000).toFixed(1)} s in your browser · seed ${cfg.seed}</p>
    <div class="table-wrap"><table>
      <thead><tr><th>#</th><th>Agent</th><th class="num">Avg score / hole</th><th></th><th class="num">Win rate</th><th class="num">Gap</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="muted small">Lower is better. ± is the 95% confidence interval of the per-match average. Win rate is per seat; ties split the win.</p>
    <div class="btn-row"><button class="btn" id="copy">Copy as Markdown</button></div>`;
  $("#copy", view).addEventListener("click", async (e) => {
    try { await navigator.clipboard.writeText(md); e.target.textContent = "Copied"; }
    catch { e.target.textContent = "Copy failed"; }
  });
}
