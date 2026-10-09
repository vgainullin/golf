// Play a match against AI opponents.

import {
  Hole, TAKE_FACE, DRAW_DECK, placeAction, flipAction, isPlace, actionPos,
  cardLabel, makeRng, winShares,
} from "./engine.js";
import { AGENTS, makeAgent } from "./agents.js";
import { $, esc, cardHTML, matchedSlots } from "./ui.js";
import { modelLink } from "./pages.js";

const SPEEDS = { slow: 1100, normal: 600, fast: 220 };
const DEFAULT_OPPONENTS = ["improved", "lookahead", "heuristic"];
const PLAYABLE = ["lookahead", "improved", "heuristic", "simple", "random"];

let session = 0; // bumps on every render so stale timers from a previous visit stop

export function renderPlay(view, params) {
  const my = ++session;
  const preset = params.get("vs");
  const opps = preset && AGENTS[preset] ? [preset, preset, preset] : DEFAULT_OPPONENTS;

  const opt = (sel) => PLAYABLE.map((k) => `<option value="${k}" ${k === sel ? "selected" : ""}>${AGENTS[k].name}</option>`).join("");
  view.innerHTML = `
    <div class="eyebrow">Play</div>
    <h1>Beat the bots at Golf</h1>
    <p class="lede">You sit in seat 1 against three AI players. Lowest total after the last hole wins. Not sure what to do? Ask the Bayes coach for a hint, or read <a href="#/how-to-play">how to play</a> and <a href="#/models">who you're up against</a>.</p>
    <details class="panel" style="margin-bottom:1rem">
      <summary><b>How to play</b></summary>
      <ul class="small">
        <li>Everyone has six face-down cards in two rows of three. The hole ends when someone turns their last card face-up; everyone else then gets one more turn.</li>
        <li>On your turn, take the face-up discard or draw from the deck. Then either put the card on one of your six slots (the old card is discarded face-up), or discard it and flip one of your face-down cards.</li>
        <li>Scores: 2 = −2, A = 1, K = 0, 3–9 face value, 10/J/Q = 10. Two cards of the same rank in a column cancel to 0.</li>
        <li><a href="#/how-to-play">Full rules with examples</a></li>
      </ul>
    </details>
    <div class="panel" id="setup">
      <div class="form-row">
        <label class="field">Seat 2<select id="o1">${opt(opps[0])}</select></label>
        <label class="field">Seat 3<select id="o2">${opt(opps[1])}</select></label>
        <label class="field">Seat 4<select id="o3">${opt(opps[2])}</select></label>
        <label class="field">Holes<select id="holes"><option>1</option><option selected>3</option><option>9</option></select></label>
        <label class="field">AI speed<select id="speed"><option value="slow">Slow</option><option value="normal" selected>Normal</option><option value="fast">Fast</option></select></label>
        <button class="btn primary" id="start">Deal</button>
      </div>
    </div>
    <div id="table" style="margin-top:1rem"></div>
    <div id="board" style="margin-top:1rem"></div>`;

  $("#start", view).addEventListener("click", () => {
    const keys = ["#o1", "#o2", "#o3"].map((s) => $(s, view).value);
    startMatch(view, my, keys, +$("#holes", view).value, () => SPEEDS[$("#speed", view).value]);
  });
}

function startMatch(view, my, oppKeys, holes, speed) {
  const rng = makeRng((Math.random() * 2 ** 32) >>> 0);
  const seats = [{ name: "You", human: true }, ...oppKeys.map((k) => ({ name: AGENTS[k].name, key: k }))];
  const agents = [null, ...oppKeys.map((k) => makeAgent(k, rng))];
  const coach = makeAgent("lookahead", rng);
  const m = { seats, agents, coach, holes, holeIdx: 0, history: [], hole: null, mode: null, hint: null, lastMove: "" };
  const alive = () => my === session;
  const tableEl = $("#table", view);
  const boardEl = $("#board", view);

  const observeAll = () => {
    m.agents.forEach((a, p) => a && a.observe && a.observe(m.hole, p));
    m.coach.observe(m.hole, 0);
  };

  function startHole() {
    m.hole = new Hole(4, rng);
    m.agents.forEach((a) => a && a.resetHole && a.resetHole());
    m.coach.resetHole();
    observeAll();
    m.mode = null; m.hint = null; m.lastMove = "";
    render();
    loop();
  }

  function describe(p, stage) {
    const h = m.hole, e = h.log[h.log.length - 1], who = m.seats[p].name;
    if (stage === 0) return e.action === TAKE_FACE ? `${who} took ${cardLabel(e.card)} from the discard.` : `${who} drew from the deck.`;
    if (isPlace(e.action)) return `${who} placed ${cardLabel(e.card)} and discarded ${cardLabel(e.discarded)}.`;
    return `${who} discarded ${cardLabel(e.card)} and flipped a card.`;
  }

  function loop() {
    if (!alive()) return;
    const h = m.hole;
    if (h.done) return endHole();
    const p = h.current;
    if (m.seats[p].human) { render(); return; }
    const delay = speed();
    setTimeout(() => {
      if (!alive()) return;
      h.step0(m.agents[p].stage0(h, p));
      observeAll();
      m.lastMove = describe(p, 0);
      render();
      setTimeout(() => {
        if (!alive()) return;
        h.step1(m.agents[p].stage1(h, p));
        if (!h.done) observeAll();
        m.lastMove = describe(p, 1);
        render();
        loop();
      }, delay);
    }, delay);
  }

  function humanAct(action) {
    const h = m.hole;
    m.hint = null;
    if (h.stage === 0) {
      h.step0(action);
      observeAll();
      m.lastMove = describe(0, 0);
      m.mode = null;
      render();
    } else {
      h.step1(action);
      if (!h.done) observeAll();
      m.lastMove = describe(0, 1);
      m.mode = null;
      loop();
    }
  }

  function endHole() {
    const scores = m.hole.finalScores();
    m.history.push(scores);
    m.holeIdx += 1;
    render();
  }

  function totals() {
    return [0, 1, 2, 3].map((p) => m.history.reduce((s, row) => s + row[p], 0));
  }

  function render() {
    if (!alive()) return;
    const h = m.hole;
    const human = !h.done && h.current === 0;
    const tot = totals();

    const oppHTML = [1, 2, 3].map((p) => {
      const ms = matchedSlots(h.cards[p], h.revealed[p]);
      const cards = h.cards[p].map((c, i) => cardHTML(c, { faceUp: h.revealed[p][i], small: true, cls: ms.has(i) ? "match" : "" })).join("");
      return `<div class="opp ${!h.done && h.current === p ? "turn" : ""}">
        <div class="who"><b>${modelLink(m.seats[p].key, m.seats[p].name, { newTab: true })}</b><span>showing ${h.visibleScore(p)} · total ${tot[p]}</span></div>
        <div class="layout">${cards}</div></div>`;
    }).join("");

    const canDiscard = human && h.stage === 1 && h.revealed[0].some((r) => !r);
    const hintAct = m.hint;
    const deckCls = human && h.stage === 0 ? "clickable" + (hintAct === DRAW_DECK ? " hint" : "") : "";
    const pileCls = human && (h.stage === 0 || canDiscard) ? "clickable" + (hintAct === TAKE_FACE || (hintAct != null && hintAct >= 9) ? " hint" : "") : "";
    const held = h.holding[0];

    const ms0 = matchedSlots(h.cards[0], h.revealed[0]);
    const mine = h.cards[0].map((c, i) => {
      let clickable = false;
      if (human && h.stage === 1) clickable = m.mode === "flip" ? !h.revealed[0][i] : true;
      const hinted = hintAct != null && h.stage === 1 && actionPos(hintAct) === i && (m.mode === "flip" ? hintAct >= 9 : hintAct < 9);
      const cls = [clickable ? "clickable" : "", ms0.has(i) ? "match" : "", hinted ? "hint" : ""].join(" ");
      return cardHTML(c, { faceUp: h.revealed[0][i], cls, attrs: `data-slot="${i}" role="button" tabindex="${clickable ? 0 : -1}"` });
    }).join("");

    let prompt = "";
    if (h.done) prompt = "Hole over.";
    else if (!human) prompt = `${esc(m.seats[h.current].name)} is thinking…`;
    else if (h.stage === 0) prompt = "Your turn: take the discard or draw from the deck.";
    else if (m.mode === "flip") prompt = "Pick a face-down card to flip.";
    else prompt = `Place ${cardLabel(held)} on a slot${canDiscard ? ", or click the discard pile to throw it away and flip a card" : ""}.`;

    tableEl.innerHTML = `
      <div class="felt">
        <div class="opps">${oppHTML}</div>
        <div class="center">
          <div class="pile"><div data-act="deck" class="${deckCls}" role="button" tabindex="0">${cardHTML(0, { faceUp: false, cls: deckCls })}</div><span>Deck · ${h.deckRemaining}</span></div>
          <div class="pile"><div data-act="pile" role="button" tabindex="0">${cardHTML(h.pile.length ? h.discardTop : -1, { cls: pileCls })}</div><span>Discard</span></div>
          <div class="pile"><div>${held >= 0 ? cardHTML(held, { cls: "just" }) : cardHTML(-1)}</div><span>In hand</span></div>
        </div>
        <div class="me ${human ? "turn" : ""}">
          <div><div class="who"><b>You</b><span>showing ${h.visibleScore(0)} · total ${tot[0]}</span></div><div class="layout">${mine}</div></div>
        </div>
        <div class="prompt">${prompt}</div>
        <div class="last-move">${esc(m.lastMove)}</div>
        <div class="btn-row" style="justify-content:center;margin-top:0.6rem">
          ${human ? `<button class="btn" data-act="hint">Ask the coach</button>` : ""}
          ${m.mode === "flip" ? `<button class="btn" data-act="cancel">Cancel</button>` : ""}
          ${h.done && m.holeIdx < m.holes ? `<button class="btn primary" data-act="next">Deal hole ${m.holeIdx + 1}</button>` : ""}
          ${h.done && m.holeIdx >= m.holes ? `<button class="btn primary" data-act="again">Play again</button>` : ""}
        </div>
      </div>`;

    boardEl.innerHTML = scoreboard();
  }

  function scoreboard() {
    if (!m.history.length) return "";
    const tot = totals();
    const finished = m.holeIdx >= m.holes && m.hole.done;
    const shares = winShares(tot);
    const rows = m.seats.map((s, p) => `<tr class="${p === 0 ? "me-row" : ""}"><td>${s.human ? "You" : modelLink(s.key, s.name, { newTab: true })}</td>${m.history.map((r) => `<td class="num">${r[p]}</td>`).join("")}<td class="num"><b>${tot[p]}</b></td></tr>`).join("");
    let banner = "";
    if (finished) {
      const msg = shares[0] === 1 ? "You won the match. Nice golf." : shares[0] > 0 ? "You tied for the win." : `${esc(m.seats[shares.indexOf(Math.max(...shares))].name)} won this one.`;
      banner = `<p><b>${msg}</b></p>`;
    }
    return `<div class="panel">${banner}<div class="table-wrap"><table class="scoreboard"><thead><tr><th>Player</th>${m.history.map((_, i) => `<th class="num">Hole ${i + 1}</th>`).join("")}<th class="num">Total</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }

  tableEl.onclick = (ev) => {
    const h = m.hole;
    const actEl = ev.target.closest("[data-act]");
    const slotEl = ev.target.closest("[data-slot]");
    const human = !h.done && h.current === 0;
    if (actEl) {
      const act = actEl.dataset.act;
      if (act === "next") return startHole();
      if (act === "again") return renderPlay(view, new URLSearchParams());
      if (act === "cancel") { m.mode = null; return render(); }
      if (!human) return;
      if (act === "hint") {
        m.hint = h.stage === 0 ? m.coach.stage0(h, 0) : m.coach.stage1(h, 0);
        if (h.stage === 1 && m.hint >= 9) m.mode = "flip";
        m.lastMove = coachText(h, m.hint);
        return render();
      }
      if (h.stage === 0 && act === "deck") return humanAct(DRAW_DECK);
      if (h.stage === 0 && act === "pile") return humanAct(TAKE_FACE);
      if (h.stage === 1 && act === "pile" && h.revealed[0].some((r) => !r)) { m.mode = "flip"; return render(); }
      return;
    }
    if (slotEl && human && h.stage === 1) {
      const i = +slotEl.dataset.slot;
      if (m.mode === "flip") { if (!h.revealed[0][i]) humanAct(flipAction(i)); }
      else humanAct(placeAction(i));
    }
  };
  tableEl.onkeydown = (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); ev.target.click(); } };

  $("#setup", view).style.display = "none";
  startHole();
}

function coachText(h, a) {
  const slot = (i) => `${i < 3 ? "top" : "bottom"} row, column ${(i % 3) + 1}`;
  if (h.stage === 0) return a === TAKE_FACE ? `Coach: take the ${cardLabel(h.discardTop)}.` : "Coach: draw from the deck.";
  if (isPlace(a)) return `Coach: place it on the ${slot(actionPos(a))}.`;
  return `Coach: discard it and flip the ${slot(actionPos(a))}.`;
}
