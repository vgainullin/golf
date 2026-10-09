// Router and the leaderboard, research and funding views.

import { renderPlay } from "./play.js";
import { renderArena } from "./arena.js";
import { renderHowTo, renderModels, modelLink } from "./pages.js";
import { $, esc, loadJSON, fmt, pct, money, REPO_BASE } from "./ui.js";

const view = $("#view");

const ROUTES = {
  "": renderHome,
  "how-to-play": renderHowTo,
  play: renderPlay,
  models: renderModels,
  arena: renderArena,
  research: renderResearch,
  fund: renderFund,
};

function route() {
  const hash = location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = hash.split("?");
  const [name, ...rest] = path.split("/");
  const fn = ROUTES[name] || renderHome;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === (ROUTES[name] ? name : "")));
  const params = new URLSearchParams(query);
  Promise.resolve(fn(view, params, rest)).catch((err) => {
    view.innerHTML = `<div class="panel"><h3>Something went wrong</h3><p class="muted">${esc(err.message)}</p></div>`;
  });
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);
route();

// ---------------------------------------------------------------------------
// Leaderboard (home)
// ---------------------------------------------------------------------------

async function renderHome(view) {
  const lb = await loadJSON("data/leaderboard.json");
  const main = lb.rows.filter((r) => !r.reference).sort((a, b) => a.avg - b.avg);
  const refs = lb.rows.filter((r) => r.reference);
  const best = main[0];
  const matches = Math.max(...main.map((r) => r.games));

  // "=1"-style ties are not computed here: the seat-cycled table has no CIs.
  const row = (r, i) => `
    <tr class="${r.reference ? "ref" : ""}">
      <td class="rank">${r.reference ? "–" : i + 1}</td>
      <td><b>${modelLink(r.model, r.agent)}</b> <span class="muted mono small">${esc(r.code)}</span></td>
      <td><span class="pill">${esc(r.kind)}</span></td>
      <td class="num">${fmt(r.avg)}</td>
      <td class="num">${pct(r.win)}</td>
      <td class="num">${r.games.toLocaleString()}</td>
      <td class="small"><a href="${REPO_BASE}${esc(r.source)}" target="_blank" rel="noopener">${esc(r.config)}</a></td>
      <td>${r.playable ? `<a class="btn" href="#/play?vs=${esc(r.playable)}">Play</a>` : `<span class="muted small">Python only</span>`}</td>
    </tr>`;

  view.innerHTML = `
    <section class="hero">
      <div class="eyebrow">Golf Agents Lab · open research</div>
      <h1>AI agents play Golf, the card game</h1>
      <p class="lede">Hand-coded heuristics, deep RL, a Bayesian card counter and language models sit at the same table for thousands of seat-cycled matches. Play them yourself, run your own competitions, read the lab notes, and fund the next experiment.</p>
      <div class="btn-row">
        <a class="btn primary" href="#/play">Play against the AI</a>
        <a class="btn" href="#/arena">Run a competition</a>
        <a class="btn" href="#/fund">Fund the research</a>
      </div>
      <div class="stats">
        <div class="stat"><div class="k">Best avg / hole</div><div class="v">${fmt(best.avg)}</div></div>
        <div class="stat"><div class="k">Leader</div><div class="v" style="font-size:1rem;font-family:var(--sans)">${esc(best.agent)}</div></div>
        <div class="stat"><div class="k">Matches per agent</div><div class="v">${matches.toLocaleString()}</div></div>
        <div class="stat"><div class="k">Mode</div><div class="v" style="font-size:1rem;font-family:var(--sans)">4 players · 9 holes</div></div>
      </div>
    </section>

    <h2>Leaderboard</h2>
    <p class="muted small">${esc(lb.note)} Lower score is better. Updated ${esc(lb.updated)}.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>#</th><th>Agent</th><th>Type</th><th class="num">Avg / hole</th><th class="num">Win rate</th><th class="num">Matches</th><th>Matchup</th><th></th></tr></thead>
      <tbody>${main.map(row).join("")}${refs.map(row).join("")}</tbody>
    </table></div>

    <h2>Benchmark your own agent</h2>
    <div class="grid-3">
      <div class="panel"><div class="step-num">01</div><h3>Write a player</h3><p class="muted small">Implement a <code>stage0</code> (take or draw) and <code>stage1</code> (place or flip) function against the vectorized simulator in <code>src/vectorized_golf.py</code>, or point the LLM harness at your model.</p></div>
      <div class="panel"><div class="step-num">02</div><h3>Run seat cycling</h3><p class="muted small"><code>uv run python -m scripts.seat_cycling --roster L,D,I,R</code> plays every seating so no agent benefits from acting first.</p></div>
      <div class="panel"><div class="step-num">03</div><h3>Send a pull request</h3><p class="muted small">Add your result and write-up under <code>data/</code> or <code>docs/</code>. Merged results appear on this leaderboard and in Research.</p></div>
    </div>

    <h2>Method</h2>
    <dl class="method panel">
      <dt>Games</dt><dd>Four players, six cards each in a 2×3 grid, nine holes. Lowest cumulative score wins; ties split the win.</dd>
      <dt>Decisions</dt><dd>Each turn is two decisions: take the discard or draw, then place the card or discard it and flip a face-down card.</dd>
      <dt>Scoring</dt><dd>2 = −2, A = 1, K = 0, 3–9 face value, 10/J/Q = 10. A matching pair in a column scores 0.</dd>
      <dt>Seat cycling</dt><dd>Every distinct seating of the roster plays the same number of matches, so results are free of seat-order bias.</dd>
      <dt>Checks</dt><dd>MDP diagnostics run before training, and every evaluation reports behavioral metrics (column matches, take rate, swaps of face-up cards) alongside the score.</dd>
    </dl>`;
}

// ---------------------------------------------------------------------------
// Research notes
// ---------------------------------------------------------------------------

async function renderResearch(view, params, rest) {
  const [{ notes }, funding] = await Promise.all([loadJSON("data/notes.json"), loadJSON("data/funding.json")]);
  const current = notes.find((n) => n.id === rest[0]);

  const agenda = funding.projects.map((p) => `
    <div class="panel project">
      <div class="head"><h3>${esc(p.title)}</h3><span class="pill ${p.status === "done" ? "ok" : ""}">${esc(p.status)}</span></div>
      <p class="muted small" style="margin:0">${esc(p.summary)}</p>
      ${p.note ? `<a class="small" href="#/research/${esc(p.note)}">Read the write-up</a>` : `<a class="small" href="#/fund">Fund this</a>`}
    </div>`).join("");

  view.innerHTML = `
    <div class="eyebrow">Research</div>
    <h1>Research notes</h1>
    <p class="lede">Every experiment is written up in the repository, failures included. Notes render straight from the Markdown files, so they are always current.</p>
    <div class="notes-layout">
      <nav class="note-list">
        ${notes.map((n) => `<a class="note-link ${current && current.id === n.id ? "active" : ""}" href="#/research/${esc(n.id)}"><div class="t">${esc(n.title)}</div><div class="muted small">${esc(n.tags.join(" · "))}</div></a>`).join("")}
      </nav>
      <article class="prose" id="note">${current ? '<p class="muted">Loading…</p>' : `
        <h2 style="margin-top:0;border:0;padding:0">Research agenda</h2>
        <p class="muted">What's next, funded by the community. Agentic runs propose an experiment, train and evaluate it with the repo's scripts, and land a write-up here.</p>
        <div class="grid-2">${agenda}</div>
        <h2>All notes</h2>
        ${notes.map((n) => `<p><a href="#/research/${esc(n.id)}"><b>${esc(n.title)}</b></a><br><span class="muted small">${esc(n.summary)}</span></p>`).join("")}`}
      </article>
    </div>`;

  if (!current) return;
  const note = $("#note", view);
  const url = REPO_BASE + current.path;
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) {
    note.innerHTML = `<p class="muted">Couldn't load <code>${esc(current.path)}</code> (HTTP ${res.status}). Serve the repository root so <code>web/</code> can reach it; see <code>web/README.md</code>.</p>`;
    return;
  }
  const md = await res.text();
  const dir = url.slice(0, url.lastIndexOf("/") + 1);
  if (window.marked) {
    note.innerHTML = window.marked.parse(md);
  } else {
    note.innerHTML = `<pre style="white-space:pre-wrap">${esc(md)}</pre>`;
  }
  // Resolve relative image and link paths against the note's own directory.
  note.querySelectorAll("img[src]").forEach((img) => {
    const src = img.getAttribute("src");
    if (!/^([a-z]+:|\/|#)/i.test(src)) img.src = new URL(src, new URL(dir, location.href)).href;
  });
  note.querySelectorAll("a[href]").forEach((a) => {
    const href = a.getAttribute("href");
    if (href.startsWith("#")) a.removeAttribute("href");
    else if (!/^([a-z]+:|\/)/i.test(href)) { a.href = new URL(href, new URL(dir, location.href)).href; a.target = "_blank"; }
  });
  document.title = `${current.title} · Golf Agents Lab`;
}

// ---------------------------------------------------------------------------
// Funding
// ---------------------------------------------------------------------------

async function renderFund(view) {
  const f = await loadJSON("data/funding.json");
  const live = Boolean(f.provider.donate_url);
  const open = f.projects.filter((p) => p.status !== "done");
  const goal = open.reduce((s, p) => s + p.goal, 0);
  const raised = open.reduce((s, p) => s + p.raised, 0);

  const donateHref = (amount, project) => {
    if (!live) return null;
    const u = new URL(f.provider.donate_url);
    if (f.provider.amount_param && amount) u.searchParams.set(f.provider.amount_param, String(amount));
    if (project) u.searchParams.set("client_reference_id", project);
    return u.href;
  };
  const donateBtn = (label, amount, project, primary = false) => {
    const href = donateHref(amount, project);
    return href
      ? `<a class="btn ${primary ? "primary" : ""}" href="${esc(href)}" target="_blank" rel="noopener">${label}</a>`
      : `<button class="btn ${primary ? "primary" : ""}" disabled title="Donations aren't enabled yet">${label}</button>`;
  };

  view.innerHTML = `
    <div class="eyebrow">Fund</div>
    <h1>Crowdfund better Golf agents</h1>
    <p class="lede">Donations pay for compute and model APIs so research agents can keep running experiments, and every result is published openly in the repo and on the leaderboard.</p>
    ${live ? "" : `<p class="notice"><b>Donations aren't open yet.</b> Payments go through a hosted checkout page that hasn't been connected; the buttons switch on once a checkout link is set in <code>web/data/funding.json</code>.</p>`}

    <div class="fund-hero" style="margin-top:1.2rem">
      <div class="panel">
        <h3>Open research goals</h3>
        <div class="stats" style="grid-template-columns:repeat(2,minmax(0,1fr));margin:0.8rem 0">
          <div class="stat"><div class="k">Raised</div><div class="v">${money(raised, f.currency)}</div></div>
          <div class="stat"><div class="k">Goal</div><div class="v">${money(goal, f.currency)}</div></div>
        </div>
        <div class="progress"><span style="width:${goal ? Math.min(100, (100 * raised) / goal) : 0}%"></span></div>
        <p class="muted small">Totals are updated by hand from the payment provider's dashboard.</p>
      </div>
      <div class="panel">
        <h3>Give once</h3>
        ${f.tiers.map((t) => `<div class="tier"><div><div class="amt">${money(t.amount, f.currency)}</div><div><b>${esc(t.label)}</b> <span class="muted small">${esc(t.perk)}</span></div></div>${donateBtn("Donate", t.amount, null, true)}</div>`).join("")}
      </div>
    </div>

    <h2>Fund a specific experiment</h2>
    <div class="grid-2">
      ${open.map((p) => `
        <div class="panel project">
          <div class="head"><h3>${esc(p.title)}</h3><span class="pill">${esc(p.status)}</span></div>
          <p class="muted small" style="margin:0">${esc(p.summary)}</p>
          <p class="small" style="margin:0"><b>You get:</b> ${esc(p.deliverable)}</p>
          <div class="progress"><span style="width:${p.goal ? Math.min(100, (100 * p.raised) / p.goal) : 0}%"></span></div>
          <div class="btn-row" style="justify-content:space-between;align-items:center"><span class="mono small">${money(p.raised, f.currency)} of ${money(p.goal, f.currency)}</span>${donateBtn("Back this", null, p.id)}</div>
        </div>`).join("")}
    </div>

    <h2>Where the money goes</h2>
    <dl class="method panel">
      <dt>Compute</dt><dd>GPU time for DQN tournaments and large seat-cycled benchmarks.</dd>
      <dt>Model APIs</dt><dd>Tokens for LLM players and for the research agent that plans and writes up experiments.</dd>
      <dt>Openness</dt><dd>Code, configs, results and notes are MIT-licensed in the public repo, including experiments that didn't work.</dd>
    </dl>`;
}
