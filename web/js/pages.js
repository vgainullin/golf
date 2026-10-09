// "How to play" guide and the per-model pages.

import { RANK_LABELS, RANK_SCORES, computeScore, finalScore } from "./engine.js";
import { $, esc, loadJSON, fmt, pct, cardHTML, matchedSlots, REPO_BASE } from "./ui.js";

const card = (rank, suit = 0) => suit * 13 + rank; // rank 0 = "2", 11 = K, 12 = A

function layoutHTML(cards, revealed = [true, true, true, true, true, true]) {
  const ms = matchedSlots(cards, revealed);
  return `<div class="layout">${cards.map((c, i) => cardHTML(c, { faceUp: revealed[i], cls: ms.has(i) ? "match" : "" })).join("")}</div>`;
}

/** Link to a model's page; used wherever a model name is shown. */
export function modelLink(id, name, { newTab = false } = {}) {
  if (!id) return esc(name);
  return `<a class="model-link" href="#/models/${esc(id)}"${newTab ? ' target="_blank" rel="noopener"' : ""}>${esc(name)}</a>`;
}

// ---------------------------------------------------------------------------
// How to play
// ---------------------------------------------------------------------------

export function renderHowTo(view) {
  const exA = [card(7), card(1), card(11), card(7, 1), card(10), card(0)]; // 9 3 K / 9 Q 2
  const exB = [card(3), card(12), card(6), card(3, 2), card(12, 1), card(6, 3)]; // all columns matched
  const hiddenEx = [card(0), card(4), card(8), card(5), card(9), card(12)];
  const hiddenRev = [true, false, false, true, false, false];

  const scoreRows = [0, 12, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    .map((r) => `<tr><td>${cardHTML(card(r, r % 2 ? 1 : 0), { small: true })}</td><td>${RANK_LABELS[r]}</td><td class="num">${RANK_SCORES[r]}</td></tr>`).join("");

  view.innerHTML = `
    <div class="eyebrow">How to play</div>
    <h1>Golf in two minutes</h1>
    <p class="lede">Golf is a four-player card game where you try to end each hole with the lowest-scoring six cards. It's quick to learn and surprisingly deep, which is why it makes a good test for AI agents.</p>
    <div class="btn-row"><a class="btn primary" href="#/play">Play a hole now</a><a class="btn" href="#/models">Meet the AI players</a></div>

    <h2>1. The deal</h2>
    <div class="grid-2">
      <div class="panel">
        <p>Each player gets <b>six cards face-down</b> in two rows of three. You don't know your own cards until you flip or replace them. One card goes face-up to start the <b>discard pile</b>; the rest is the <b>deck</b>.</p>
        <p class="muted small">Some house rules flip two cards at the deal. Here every card starts face-down, which matches the repo's simulator, so web and Python results are comparable.</p>
      </div>
      <div class="panel felt" style="display:grid;place-items:center">${layoutHTML(hiddenEx, [false, false, false, false, false, false])}</div>
    </div>

    <h2>2. Your turn: two decisions</h2>
    <div class="grid-2">
      <div class="panel">
        <div class="step-num">Draw</div>
        <h3>Take the discard or draw from the deck</h3>
        <p class="muted">The discard is face-up, so you know what you're getting. The deck is a gamble. In the Play view, click the discard pile or the deck.</p>
      </div>
      <div class="panel">
        <div class="step-num">Play</div>
        <h3>Place it, or throw it away and flip</h3>
        <p class="muted">Put the card on any of your six slots; the card it replaces goes face-up onto the discard pile. Or discard it and turn one of your face-down cards face-up. In Play, click a slot to place, or click the discard pile and then a face-down card to flip.</p>
      </div>
    </div>

    <h2>3. Scoring</h2>
    <div class="grid-2">
      <div class="table-wrap"><table>
        <thead><tr><th>Card</th><th>Rank</th><th class="num">Points</th></tr></thead>
        <tbody>${scoreRows}</tbody>
      </table></div>
      <div class="project" style="display:grid;gap:0.9rem;align-content:start">
        <div class="panel">
          <h3>Columns that match score zero</h3>
          <p class="muted small">If both cards in a column have the same rank, the column scores 0, even a pair of 10s. A pair of 2s also scores 0, so you lose their −2 each.</p>
          <div class="felt" style="display:flex;gap:1.4rem;flex-wrap:wrap;justify-content:center;align-items:center">
            <div style="text-align:center">${layoutHTML(exA)}<div class="small" style="margin-top:0.4rem">9s cancel: Q + 2 + 3 + K = <b>${finalScore(exA)}</b></div></div>
            <div style="text-align:center">${layoutHTML(exB)}<div class="small" style="margin-top:0.4rem">Every column matched: <b>${finalScore(exB)}</b></div></div>
          </div>
        </div>
        <div class="panel">
          <h3>Face-down cards still count</h3>
          <p class="muted small">During play only face-up cards show in your running score. When the hole ends, everything is turned over and counted.</p>
          <div class="felt" style="display:flex;gap:1.2rem;justify-content:center;align-items:center;flex-wrap:wrap">
            ${layoutHTML(hiddenEx, hiddenRev)}
            <div class="small">Showing <b>${computeScore(hiddenEx, hiddenRev)}</b><br>Final <b>${finalScore(hiddenEx)}</b></div>
          </div>
        </div>
      </div>
    </div>

    <h2>4. Ending a hole and the match</h2>
    <div class="panel">
      <p>As soon as one player has all six cards face-up, <b>everyone else gets exactly one more turn</b>. Then all cards are revealed and scored. A match is usually nine holes; the lowest total wins, and tied players share the win.</p>
      <p class="muted small" style="margin-bottom:0">Going out early can catch opponents with high face-down cards, but it locks in your own score. Make sure it's low.</p>
    </div>

    <h2>5. Tips from the research</h2>
    <dl class="method panel">
      <dt>Keep the lows</dt><dd>2s, Kings and Aces are worth −2, 0 and 1. Taking a low discard is almost always right.</dd>
      <dt>Hunt pairs</dt><dd>A discard that matches the rank of one of your face-up cards can wipe out a whole column.</dd>
      <dt>Fix mistakes</dt><dd>Replacing a bad face-up card is allowed and often best. That one change takes the base heuristic from about 14 to about 10.5 points per hole.</dd>
      <dt>Count cards</dt><dd>Track which ranks are gone. The best agent, the <a href="#/models/lookahead">Bayes lookahead</a>, does nothing more clever than this and averages under 7 per hole against weaker players.</dd>
    </dl>
    <p class="muted small">Stuck in a game? Press <b>Ask the coach</b> on the Play page and the Bayes lookahead suggests a move for you.</p>`;
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export async function renderModels(view, params, rest) {
  const { models } = await loadJSON("data/models.json");
  const id = rest[0];
  if (!id) return renderModelIndex(view, models);
  const m = models.find((x) => x.id === id);
  if (!m) {
    view.innerHTML = `<div class="panel"><h3>No such model</h3><p class="muted">Try the <a href="#/models">model list</a>.</p></div>`;
    return;
  }
  renderModel(view, m, models);
}

function renderModelIndex(view, models) {
  view.innerHTML = `
    <div class="eyebrow">Models</div>
    <h1>The players</h1>
    <p class="lede">Every agent that has played in the repo's benchmarks, from a coin-flipper to a Bayesian card counter. Pick one to see how it works and how it scores.</p>
    <div class="grid-2">${models.map((m) => {
      const best = m.results[0];
      return `<a class="panel model-card" href="#/models/${esc(m.id)}">
        <div class="head" style="display:flex;justify-content:space-between;gap:0.6rem;align-items:start">
          <h3>${esc(m.name)} <span class="muted mono small">${esc(m.code)}</span></h3><span class="pill">${esc(m.kind)}</span>
        </div>
        <p class="muted small" style="margin:0.3rem 0 0.6rem">${esc(m.tagline)}</p>
        <div class="small">${best ? `<span class="mono"><b>${fmt(best.avg)}</b></span> <span class="muted">avg / hole · ${esc(best.config)}</span>` : `<span class="muted">No published benchmark yet</span>`}
        ${m.agent ? ` · <span class="pill ok">Playable</span>` : ""}</div>
      </a>`;
    }).join("")}</div>`;
}

function renderModel(view, m, models) {
  const rows = m.results.map((r) => `<tr>
      <td style="white-space:normal">${esc(r.config)}</td>
      <td class="num">${fmt(r.avg)}</td>
      <td class="num">${pct(r.win)}</td>
      <td class="small"><a href="${REPO_BASE}${esc(r.source)}" target="_blank" rel="noopener">${esc(r.source)}</a></td>
    </tr>`).join("");
  const idx = models.indexOf(m);
  const prev = models[(idx - 1 + models.length) % models.length];
  const next = models[(idx + 1) % models.length];
  const list = (xs) => `<ul class="small" style="margin:0;padding-left:1.1rem">${xs.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;

  view.innerHTML = `
    <div class="eyebrow"><a href="#/models">Models</a> / ${esc(m.kind)}</div>
    <h1>${esc(m.name)} <span class="muted mono" style="font-size:0.5em">${esc(m.code)}</span></h1>
    <p class="lede">${esc(m.tagline)}</p>
    <div class="btn-row">
      ${m.agent
        ? `<a class="btn primary" href="#/play?vs=${esc(m.agent)}">Play against it</a><a class="btn" href="#/arena?seat=${esc(m.agent)}">Put it in the arena</a>`
        : `<span class="notice small">Runs in Python only, so it isn't playable in the browser yet. <a href="#/fund">Fund the work</a> to bring it here.</span>`}
    </div>

    <h2>How it plays</h2>
    <div class="panel"><ol style="margin:0;padding-left:1.2rem;display:grid;gap:0.5rem">${m.how.map((h) => `<li>${esc(h)}</li>`).join("")}</ol></div>

    <div class="grid-2" style="margin-top:0.9rem">
      <div class="panel"><h3>Strengths</h3>${list(m.strengths)}</div>
      <div class="panel"><h3>Weaknesses</h3>${list(m.weaknesses)}</div>
    </div>

    <h2>Results</h2>
    ${m.results.length ? `<div class="table-wrap"><table>
      <thead><tr><th>Matchup</th><th class="num">Avg / hole</th><th class="num">Win rate</th><th>Source</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="muted small">Lower is better. Scores depend on the opponents, so only compare agents within the same matchup.</p>`
      : `<p class="muted">No published benchmark yet. ${m.agent ? `Run one in the <a href="#/arena?seat=${esc(m.agent)}">arena</a>.` : ""}</p>`}

    <h2>Code and notes</h2>
    <div class="panel small">
      <p style="margin-top:0"><b>Code:</b> ${m.code_paths.map((p) => `<a href="https://github.com/vgainullin/golf/blob/main/${esc(p)}" target="_blank" rel="noopener"><code>${esc(p)}</code></a>`).join(", ")}</p>
      ${m.notes.length ? `<p style="margin-bottom:0"><b>Read more:</b> ${m.notes.map((n) => `<a href="#/research/${esc(n.id)}">${esc(n.label)}</a>`).join(" · ")}</p>` : ""}
    </div>

    <div class="btn-row" style="justify-content:space-between;margin-top:2rem">
      <a class="btn" href="#/models/${esc(prev.id)}">← ${esc(prev.name)}</a>
      <a class="btn" href="#/models/${esc(next.id)}">${esc(next.name)} →</a>
    </div>`;
  document.title = `${m.name} · Golf Agents Lab`;
}
