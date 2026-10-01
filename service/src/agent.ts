/**
 * The research agent that maintains a project. It does two jobs per run:
 *
 *   plan()    - given the project brief, the leaderboard and what earlier runs
 *               found, decide the next experiment and write its spec.
 *   analyze() - given the result, write the report: what happened, how it
 *               compares to the baselines, and what to try next.
 *
 * With ANTHROPIC_API_KEY set it calls Claude. Without it, a scripted planner
 * and a templated reporter stand in so the loop runs offline.
 */
import Anthropic from "@anthropic-ai/sdk";
import { betaZodOutputFormat } from "@anthropic-ai/sdk/helpers/beta/zod";
import { z } from "zod";
import lookaheadSource from "../agents/lookahead.py";
import flipPartnerSource from "../agents/lookahead_flip_partner.py";
import placeMarginSource from "../agents/lookahead_place_margin.py";
import type { Env, LeaderboardRow, Project, Run, RunResult, RunSpec } from "./types";

/** The Bayes lookahead player as agent code: the fallback starting point. */
export const LOOKAHEAD_SOURCE: string = lookaheadSource;

export interface PlanContext {
  project: Project;
  leaderboard: LeaderboardRow[];
  history: Run[]; // earlier runs of this project, newest first
  ordinal: number; // how many runs this project had before this one
  start: StartingPoint; // the best entry with code; every run changes one thing in it
  donorNote?: string | null;
}

export interface StartingPoint {
  name: string;
  metric_value: number | null;
  code: string;
}

export interface Plan {
  title: string;
  hypothesis: string;
  spec: RunSpec;
}

export interface Report {
  summary: string;
  markdown: string;
}

const PlanSchema = z.object({
  title: z.string().describe("Short name for this candidate, shown on the leaderboard"),
  change: z.string().describe("The one change made to the starting point, in one sentence"),
  hypothesis: z.string().describe("Why this change should lower the score, in two or three sentences"),
  agent_code: z.string().describe("Full Python module source: the starting point with the one change applied"),
});

const ReportSchema = z.object({
  summary: z.string().describe("One sentence: the result and whether it beat the best entry"),
  findings: z.string().describe("Markdown. What the numbers say, compared with the leaderboard"),
  next_directions: z.array(z.string()).describe("Concrete experiments worth funding next"),
});

function client(env: Env): Anthropic | null {
  return env.ANTHROPIC_API_KEY ? new Anthropic({ apiKey: env.ANTHROPIC_API_KEY }) : null;
}

function describeContext(ctx: PlanContext): string {
  const lb = ctx.leaderboard
    .slice(0, 10)
    .map(
      (r, i) =>
        `${i + 1}. ${r.name}: ${r.metric_value.toFixed(3)}` +
        (r.win_rate != null ? ` (${(r.win_rate * 100).toFixed(1)}% wins)` : "") +
        (r.simulated ? " [simulated]" : "") +
        ` [${r.source}]`,
    )
    .join("\n");
  const hist = ctx.history
    .slice(0, 8)
    .map((r) => {
      const score = r.metric_value != null ? r.metric_value.toFixed(3) : r.status;
      const next = r.report_md?.split("## Next directions")[1]?.trim().slice(0, 600) ?? "";
      return `- ${r.title ?? r.id} (${score}): ${r.hypothesis ?? ""}${next ? `\n  Suggested next: ${next}` : ""}`;
    })
    .join("\n");
  return [
    `Project: ${ctx.project.name}\n${ctx.project.agent_brief}`,
    `Leaderboard on protocol ${ctx.leaderboard[0]?.protocol ?? ctx.project.protocol} (${ctx.project.metric_name}, ${
      ctx.project.lower_is_better ? "lower" : "higher"
    } is better):\n${lb || "(empty)"}`,
    `Earlier runs, newest first:\n${hist || "(none)"}`,
    `Starting point: ${ctx.start.name}${
      ctx.start.metric_value != null ? ` (${ctx.start.metric_value.toFixed(3)})` : ""
    }, the best entry with code. Its source:\n\n\`\`\`python\n${ctx.start.code}\n\`\`\``,
    ctx.donorNote ? `The donor who funded this run wrote: ${ctx.donorNote}` : "",
  ]
    .filter(Boolean)
    .join("\n\n");
}

export async function plan(env: Env, ctx: PlanContext): Promise<Plan> {
  const c = client(env);
  if (!c) return scriptedPlan(ctx);

  const response = await c.beta.messages.parse({
    model: env.AGENT_MODEL,
    max_tokens: 16000,
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    output_config: { effort: "high", format: betaZodOutputFormat(PlanSchema) },
    system:
      "You maintain an open research project. Each donation pays for exactly one run. " +
      "Every run starts from the starting point (the best entry on the leaderboard that has " +
      "code) and changes exactly one thing in it, so the result says whether that change helps. " +
      "Pick the change most likely to lower the score, using what earlier runs found; don't " +
      "repeat a change that was already tried. Return the full modified module. It must import " +
      "only torch and the repo's src modules, be vectorized over N games, and return legal actions.",
    messages: [{ role: "user", content: describeContext(ctx) + "\n\nPlan the next run." }],
  });
  if (response.stop_reason === "refusal") throw new Error("planner declined the request");
  const out = response.parsed_output;
  if (!out) throw new Error(`planner returned no parseable plan (stop_reason=${response.stop_reason})`);

  if (!out.agent_code.trim()) throw new Error("planner returned empty agent code");
  return {
    title: out.title,
    hypothesis: `Change to ${ctx.start.name}: ${out.change} ${out.hypothesis}`,
    spec: { kind: "agent_code", agent_code: out.agent_code, protocol: ctx.project.protocol, compute: "cpu" },
  };
}

export async function analyze(
  env: Env,
  ctx: PlanContext & { run: Run; result: RunResult },
): Promise<Report> {
  const c = client(env);
  if (!c) return templatedReport(ctx);

  const response = await c.beta.messages.parse({
    model: env.AGENT_MODEL,
    max_tokens: 16000,
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    output_config: { effort: "high", format: betaZodOutputFormat(ReportSchema) },
    system:
      "You write the report for one research run of an open project. Be specific and " +
      "plain: state the number, compare it with the leaderboard, say whether the " +
      "difference is meaningful given the game counts, explain what the result suggests " +
      "about the approach, and propose concrete follow-ups. No marketing tone.",
    messages: [
      {
        role: "user",
        content:
          describeContext(ctx) +
          `\n\nThis run: ${ctx.run.title}\nHypothesis: ${ctx.run.hypothesis}\nSpec:\n${
            ctx.run.spec_json
          }\n\nResult JSON:\n${JSON.stringify(trimResult(ctx.result), null, 1)}\n\nWrite the report.`,
      },
    ],
  });
  if (response.stop_reason === "refusal") throw new Error("reporter declined the request");
  const out = response.parsed_output;
  if (!out) throw new Error(`reporter returned no parseable report (stop_reason=${response.stop_reason})`);
  return { summary: out.summary, markdown: assemble(ctx, out.summary, out.findings, out.next_directions) };
}

function trimResult(r: RunResult): RunResult {
  // Training logs and tracebacks can be long; the tail is what matters.
  const copy = { ...r };
  if (typeof copy.traceback === "string") copy.traceback = copy.traceback.slice(-2000);
  const train = copy.train as { log_tail?: string } | undefined;
  if (train?.log_tail) copy.train = { ...train, log_tail: train.log_tail.slice(-2000) };
  return copy;
}

function assemble(
  ctx: { run: Run; result: RunResult; project: Project },
  summary: string,
  findings: string,
  next: string[],
): string {
  const r = ctx.result;
  const lines = [
    `# ${ctx.run.title ?? ctx.run.id}`,
    "",
    summary,
    "",
    `- Run: \`${ctx.run.id}\``,
    `- Protocol: ${r.protocol?.id ?? ctx.project.protocol}${
      r.protocol?.games_per_perm ? `, ${r.protocol.games_per_perm} games per seating, ${r.protocol.holes} holes` : ""
    }${r.protocol?.overridden ? " (game count overridden by the runner; not comparable with the main board)" : ""}`,
  ];
  if (r.metric) lines.push(`- ${r.metric.name}: **${r.metric.value.toFixed(3)}**`);
  if (r.win_rate != null) lines.push(`- Win rate: ${(r.win_rate * 100).toFixed(1)}%`);
  if (r.simulated) lines.push("- Numbers are **simulated** (mock dispatcher); nothing was executed.");
  lines.push("", "## Hypothesis", "", ctx.run.hypothesis ?? "", "", "## Findings", "", findings, "", "## Next directions", "");
  for (const n of next) lines.push(`- ${n}`);
  return lines.join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// Offline stand-ins
// ---------------------------------------------------------------------------


// Each scripted plan is the Bayes lookahead with one change. They can't build
// on whatever is on top of the board, so they always start from the lookahead.
const SCRIPTED_PLANS: { title: string; hypothesis: string; code: string }[] = [
  {
    title: "Lookahead, flip toward column matches",
    hypothesis:
      "Change to the Bayes lookahead: when it discards and flips, flip a hidden card whose column partner is already revealed instead of the first hidden slot. That flip is the only one that can complete a column match, so it should cancel more points.",
    code: flipPartnerSource,
  },
  {
    title: "Lookahead, place only on a 0.5 gain",
    hypothesis:
      "Change to the Bayes lookahead: place the held card only when it lowers expected score by at least 0.5, otherwise discard and flip. Small expected gains may not be worth giving up the information a flip reveals.",
    code: placeMarginSource,
  },
];

function scriptedPlan(ctx: PlanContext): Plan {
  const p = SCRIPTED_PLANS[ctx.ordinal % SCRIPTED_PLANS.length];
  return {
    title: p.title,
    hypothesis: p.hypothesis + " (Scripted plan: no ANTHROPIC_API_KEY configured.)",
    spec: { kind: "agent_code", agent_code: p.code, protocol: ctx.project.protocol, compute: "cpu" },
  };
}

function templatedReport(ctx: PlanContext & { run: Run; result: RunResult }): Report {
  const r = ctx.result;
  if (r.status !== "ok" || !r.metric) {
    const summary = `The run failed before producing a score: ${r.error ?? "unknown error"}.`;
    return {
      summary,
      markdown: assemble(ctx, summary, "```\n" + String(r.traceback ?? r.error ?? "").slice(-1500) + "\n```", [
        "Fix the error above and resubmit the same idea.",
      ]),
    };
  }
  const lower = !!ctx.project.lower_is_better;
  const others = ctx.leaderboard.filter((row) => row.run_id !== ctx.run.id);
  const best = others[0];
  const v = r.metric.value;
  const beats = best ? (lower ? v < best.metric_value : v > best.metric_value) : true;
  const summary = best
    ? `Scored ${v.toFixed(3)} against the best entry's ${best.metric_value.toFixed(3)} (${best.name}); ${
        beats ? "a new top score" : "it does not beat it"
      }.`
    : `Scored ${v.toFixed(3)}; this is the first entry on the leaderboard.`;
  const rank = [...others.map((o) => o.metric_value), v].sort((a, b) => (lower ? a - b : b - a)).indexOf(v) + 1;
  const opp = r.per_label
    ? Object.entries(r.per_label)
        .filter(([k]) => k !== "C")
        .map(([k, s]) => `${k} ${s.avg_score_per_hole.toFixed(3)}`)
        .join(", ")
    : "";
  const findings = [
    `Rank ${rank} of ${others.length + 1} on ${r.protocol?.id ?? ctx.project.protocol}.`,
    opp ? `Opponents in the same games: ${opp}.` : "",
    "This report is a template (no ANTHROPIC_API_KEY configured), so it states the numbers without interpreting them.",
  ]
    .filter(Boolean)
    .join(" ");
  return {
    summary,
    markdown: assemble(ctx, summary, findings, [
      "Configure ANTHROPIC_API_KEY so the agent can interpret results and plan follow-ups.",
    ]),
  };
}
