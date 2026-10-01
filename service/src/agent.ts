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
import type { Env, LeaderboardRow, Project, Run, RunResult, RunSpec } from "./types";

export interface PlanContext {
  project: Project;
  leaderboard: LeaderboardRow[];
  history: Run[]; // earlier runs of this project, newest first
  ordinal: number; // how many runs this project had before this one
  donorNote?: string | null;
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
  hypothesis: z.string().describe("What you expect and why, in two or three sentences"),
  kind: z.enum(["agent_code", "train"]),
  agent_code: z
    .string()
    .nullable()
    .describe("For kind=agent_code: full Python module source. Null otherwise."),
  train_args_json: z
    .string()
    .nullable()
    .describe("For kind=train: JSON object of src.tournament flags (snake_case). Null otherwise."),
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
      "Pick the single experiment most likely to improve the leaderboard or to teach " +
      "something the next run can use. Build on earlier findings; don't repeat a run. " +
      "Runs execute on CPU, so keep training inside the stated caps. Agent code must be " +
      "self-contained, import only torch and the repo's src modules, and be vectorized over N games.",
    messages: [{ role: "user", content: describeContext(ctx) + "\n\nPlan the next run." }],
  });
  if (response.stop_reason === "refusal") throw new Error("planner declined the request");
  const out = response.parsed_output;
  if (!out) throw new Error(`planner returned no parseable plan (stop_reason=${response.stop_reason})`);

  const spec: RunSpec = { kind: out.kind, protocol: ctx.project.protocol, compute: "cpu" };
  if (out.kind === "agent_code") {
    if (!out.agent_code) throw new Error("planner chose agent_code without code");
    spec.agent_code = out.agent_code;
  } else {
    spec.train = { args: JSON.parse(out.train_args_json ?? "{}") };
  }
  return { title: out.title, hypothesis: out.hypothesis, spec };
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


const SCRIPTED_PLANS: { title: string; hypothesis: string; spec: Omit<RunSpec, "protocol" | "compute"> }[] = [
  {
    title: "Improved heuristic as candidate",
    hypothesis:
      "Seat the improved heuristic as C to get its number on this exact protocol, so later code agents have a floor to beat.",
    spec: {
      kind: "agent_code",
      agent_code: [
        "from src.vectorized_golf import heuristic_stage0, improved_stage1",
        "",
        "def stage0(state, seat):",
        "    return heuristic_stage0(state, seat)",
        "",
        "def stage1(state, seat):",
        "    return improved_stage1(state, seat)",
        "",
      ].join("\n"),
    },
  },
  {
    title: "Lookahead draw + improved placement",
    hypothesis:
      "Most of the lookahead player's edge may come from the draw decision. Pair its belief-based stage 0 with the cheap improved-heuristic stage 1 and see how much of the gap closes.",
    spec: {
      kind: "agent_code",
      agent_code: [
        "from src.bayes_optimal import BayesBeliefTracker, lookahead_stage0",
        "from src.vectorized_golf import improved_stage1",
        "",
        "_tracker = None",
        "",
        "def reset(state, seat):",
        "    global _tracker",
        "    _tracker = BayesBeliefTracker(state.player_cards.shape[0], state.player_cards.device)",
        "    _tracker.observe(state, my_player_id=seat)",
        "",
        "def observe(state, seat):",
        "    _tracker.observe(state, my_player_id=seat)",
        "",
        "def stage0(state, seat):",
        "    _tracker.observe(state, my_player_id=seat)",
        "    return lookahead_stage0(state, seat, _tracker)",
        "",
        "def stage1(state, seat):",
        "    return improved_stage1(state, seat)",
        "",
      ].join("\n"),
    },
  },
  {
    title: "Small v3 DQN with win bonus (CPU)",
    hypothesis:
      "A CPU-sized version of the Exp14 recipe (v3 model, hindsight shaping, win bonus 0.3) shows how much of the champion's strength survives a 30x smaller training budget.",
    spec: {
      kind: "train",
      train: {
        args: {
          model_variant: "v3",
          hidden_dim_choices: [256],
          embedding_dim: 64,
          population_size: 4,
          generations: 10,
          episodes_per_gen: 500,
          buffer_capacity: 50000,
          batch_size: 256,
          reward_shaping: "hindsight",
          win_bonus: 0.3,
        },
      },
    },
  },
];

function scriptedPlan(ctx: PlanContext): Plan {
  const p = SCRIPTED_PLANS[ctx.ordinal % SCRIPTED_PLANS.length];
  return {
    title: p.title,
    hypothesis: p.hypothesis + " (Scripted plan: no ANTHROPIC_API_KEY configured.)",
    spec: { ...p.spec, protocol: ctx.project.protocol, compute: "cpu" },
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
