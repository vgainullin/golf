/**
 * Dispatchers hand a planned run to something that can execute it. The Worker
 * never runs training or evaluation itself.
 */
import type { Env, Run, RunResult, RunSpec } from "./types";

export interface DispatchOutcome {
  external_ref: string | null;
  /** Set when the dispatcher finished the run synchronously (mock). */
  result?: RunResult;
}

export async function dispatch(env: Env, run: Run, spec: RunSpec): Promise<DispatchOutcome> {
  if (spec.compute === "gpu" && env.ALLOW_GPU !== "true") {
    throw new Error("GPU compute is disabled (ALLOW_GPU is not 'true'); GPU runs launch paid instances");
  }
  switch (env.DISPATCHER) {
    case "mock":
      return { external_ref: "mock", result: await simulate(run, spec) };
    case "queue":
      // A runner picks it up via POST /v1/runner/claim.
      return { external_ref: null };
    case "github":
      return { external_ref: await dispatchGithub(env, run, spec) };
    default:
      throw new Error(`unknown DISPATCHER ${env.DISPATCHER}`);
  }
}

/**
 * Deterministic fake numbers so the loop (leaderboard, report, next plan) can be
 * exercised without compute. Rows it produces are flagged simulated.
 */
async function simulate(run: Run, spec: RunSpec): Promise<RunResult> {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(run.id + JSON.stringify(spec))),
  );
  const u = (digest[0] * 256 + digest[1]) / 65536; // [0, 1)
  const value = 9.2 + u * 3.2; // somewhere between "close to lookahead" and "heuristic"
  const win = Math.max(0, 0.45 - (value - 9.2) * 0.12);
  return {
    status: "ok",
    simulated: true,
    protocol: { id: spec.protocol, games_per_perm: 0, holes: 0 },
    metric: { name: "avg_score_per_hole", value: Math.round(value * 1000) / 1000, lower_is_better: true },
    win_rate: Math.round(win * 1000) / 1000,
  };
}

async function dispatchGithub(env: Env, run: Run, spec: RunSpec): Promise<string> {
  if (!env.GITHUB_TOKEN) throw new Error("GITHUB_TOKEN is not set");
  if (!env.PUBLIC_URL) throw new Error("PUBLIC_URL is not set; the workflow needs it to fetch the spec and report back");
  const url = `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/research_run.yml/dispatches`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "research-runs",
    },
    body: JSON.stringify({
      ref: env.GITHUB_REF,
      inputs: { run_id: run.id, api_url: env.PUBLIC_URL, compute: spec.compute },
    }),
  });
  if (res.status !== 204) {
    throw new Error(`workflow dispatch failed: HTTP ${res.status} ${await res.text()}`);
  }
  // workflow_dispatch returns no run id; the workflow reports its URL via /started.
  return `github:${env.GITHUB_REPO}`;
}
