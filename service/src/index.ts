/**
 * Research-runs API (Cloudflare Worker).
 *
 * Public:
 *   GET  /v1/projects
 *   GET  /v1/projects/:slug
 *   POST /v1/projects/:slug/donations        {donor, amount_usd | amount_cents, note?, submission?}
 *   GET  /v1/projects/:slug/leaderboard      ?protocol=
 *   GET  /v1/projects/:slug/runs
 *   GET  /v1/runs/:id                         run + events
 *   GET  /v1/runs/:id/report                  markdown
 * Runner (Bearer RUNNER_TOKEN):
 *   POST /v1/runner/claim                     {runner?} -> next queued run, or 204
 *   GET  /v1/runner/runs/:id/spec
 *   POST /v1/runner/runs/:id/started          {external_ref?}
 *   POST /v1/runner/runs/:id/result           RunResult
 * Admin (Bearer ADMIN_TOKEN):
 *   POST /v1/admin/tick                       advance every runnable run now
 *   POST /v1/admin/projects/:slug/baselines   {label, name} score an existing player as a baseline
 */
import {
  getProject,
  getRun,
  leaderboard,
  listProjects,
  listRuns,
  runEvents,
  transition,
} from "./db";
import { donate, drive, HttpError, queueBaseline, recordResult, tick, type Submission } from "./pipeline";
import type { Env, Project, Run, RunResult } from "./types";

type Handler = (req: Request, env: Env, ctx: ExecutionContext, params: string[]) => Promise<Response>;

const routes: [string, RegExp, Handler][] = [
  ["GET", /^\/v1\/projects$/, async (_r, env) => json({ projects: (await listProjects(env)).map(publicProject) })],
  ["GET", /^\/v1\/projects\/([\w-]+)$/, async (_r, env, _c, [slug]) => {
    const p = await mustProject(env, slug);
    const lb = await leaderboard(env, p);
    const counts = await env.DB.prepare("SELECT status, COUNT(*) AS n FROM runs WHERE project_id = ? GROUP BY status")
      .bind(p.id)
      .all<{ status: string; n: number }>();
    return json({
      project: publicProject(p),
      best: lb[0] ?? null,
      runs_by_status: Object.fromEntries(counts.results.map((c) => [c.status, c.n])),
    });
  }],
  ["POST", /^\/v1\/projects\/([\w-]+)\/donations$/, async (req, env, ctx, [slug]) => {
    const p = await mustProject(env, slug);
    const body = await readJson<{
      donor?: string;
      amount_usd?: number;
      amount_cents?: number;
      note?: string;
      submission?: Submission;
    }>(req);
    const cents = body.amount_cents ?? Math.round((body.amount_usd ?? 0) * 100);
    if (!Number.isInteger(cents) || cents < 100 || cents > 1_000_000) {
      throw new HttpError(400, "amount must be between $1 and $10,000");
    }
    const donor = (body.donor ?? "anonymous").slice(0, 80);
    const out = await donate(env, p, donor, cents, body.note?.slice(0, 2000) ?? null, body.submission ?? null);
    // Kick the new runs forward without making the donor wait. Planning can
    // take longer than waitUntil allows; the cron picks up anything left.
    for (const id of out.runs) ctx.waitUntil(drive(env, id).catch(() => undefined));
    return json(
      {
        ...out,
        run_price_cents: p.run_price_cents,
        message: out.runs.length
          ? `Funded ${out.runs.length} run(s).`
          : `Added to the pool; ${p.run_price_cents - (out.balance_cents ?? 0)} cents until the next run.`,
      },
      201,
    );
  }],
  ["GET", /^\/v1\/projects\/([\w-]+)\/leaderboard$/, async (req, env, _c, [slug]) => {
    const p = await mustProject(env, slug);
    const protocol = new URL(req.url).searchParams.get("protocol") ?? p.protocol;
    const rows = await leaderboard(env, p, protocol);
    const protocols = await env.DB.prepare("SELECT DISTINCT protocol FROM leaderboard WHERE project_id = ?")
      .bind(p.id)
      .all<{ protocol: string }>();
    return json({
      project: p.slug,
      protocol,
      metric: p.metric_name,
      lower_is_better: !!p.lower_is_better,
      protocols: protocols.results.map((r) => r.protocol),
      rows: rows.map((r, i) => ({ rank: i + 1, ...r, simulated: !!r.simulated })),
    });
  }],
  ["GET", /^\/v1\/projects\/([\w-]+)\/runs$/, async (_r, env, _c, [slug]) => {
    const p = await mustProject(env, slug);
    return json({ runs: (await listRuns(env, p.id)).map(publicRun) });
  }],
  ["GET", /^\/v1\/runs\/([\w-]+)$/, async (_r, env, _c, [id]) => {
    const run = await mustRun(env, id);
    return json({
      run: {
        ...publicRun(run),
        spec: run.spec_json ? JSON.parse(run.spec_json) : null,
        result: run.result_json ? JSON.parse(run.result_json) : null,
        report_md: run.report_md,
        plan_review: run.plan_review,
      },
      events: await runEvents(env, id),
    });
  }],
  ["GET", /^\/v1\/runs\/([\w-]+)\/report$/, async (_r, env, _c, [id]) => {
    const run = await mustRun(env, id);
    if (!run.report_md) throw new HttpError(404, `no report yet; run is ${run.status}`);
    return new Response(run.report_md, { headers: { "content-type": "text/markdown; charset=utf-8" } });
  }],

  // --- runner ---
  ["POST", /^\/v1\/runner\/claim$/, async (req, env) => {
    auth(req, env.RUNNER_TOKEN, "RUNNER_TOKEN");
    const body = await readJson<{ runner?: string }>(req, true);
    const runner = (body.runner ?? "runner").slice(0, 60);
    for (let i = 0; i < 5; i++) {
      const run = await env.DB.prepare(
        "SELECT * FROM runs WHERE status = 'dispatched' AND dispatcher = 'queue' ORDER BY updated_at, rowid LIMIT 1",
      ).first<Run>();
      if (!run) return new Response(null, { status: 204 });
      if (await transition(env, run.id, "dispatched", "running", { external_ref: `runner:${runner}` })) {
        return json({ run_id: run.id, spec: JSON.parse(run.spec_json!) });
      }
    }
    return new Response(null, { status: 204 });
  }],
  ["GET", /^\/v1\/runner\/runs\/([\w-]+)\/spec$/, async (req, env, _c, [id]) => {
    auth(req, env.RUNNER_TOKEN, "RUNNER_TOKEN");
    const run = await mustRun(env, id);
    if (!run.spec_json) throw new HttpError(409, `run is ${run.status} and has no spec`);
    return json(JSON.parse(run.spec_json));
  }],
  ["POST", /^\/v1\/runner\/runs\/([\w-]+)\/started$/, async (req, env, _c, [id]) => {
    auth(req, env.RUNNER_TOKEN, "RUNNER_TOKEN");
    const body = await readJson<{ external_ref?: string }>(req, true);
    const fields = body.external_ref ? { external_ref: body.external_ref.slice(0, 500) } : {};
    if (!(await transition(env, id, "dispatched", "running", fields))) {
      throw new HttpError(409, `run is ${(await mustRun(env, id)).status}`);
    }
    return json({ ok: true });
  }],
  ["POST", /^\/v1\/runner\/runs\/([\w-]+)\/result$/, async (req, env, ctx, [id]) => {
    auth(req, env.RUNNER_TOKEN, "RUNNER_TOKEN");
    const result = await readJson<RunResult>(req);
    if (result.status !== "ok" && result.status !== "error") throw new HttpError(400, "result.status must be ok or error");
    if (result.simulated) throw new HttpError(400, "runners can't post simulated results");
    const run = await recordResult(env, id, result);
    ctx.waitUntil(drive(env, id).catch(() => undefined));
    return json({ run: publicRun(run) });
  }],

  // --- admin ---
  ["POST", /^\/v1\/admin\/tick$/, async (req, env) => {
    auth(req, env.ADMIN_TOKEN, "ADMIN_TOKEN");
    return json(await tick(env));
  }],
  ["POST", /^\/v1\/admin\/projects\/([\w-]+)\/baselines$/, async (req, env, ctx, [slug]) => {
    auth(req, env.ADMIN_TOKEN, "ADMIN_TOKEN");
    const p = await mustProject(env, slug);
    const body = await readJson<{ label?: string; name?: string }>(req);
    if (!body.label || !/^[A-Z][A-Z0-9]{0,3}$/.test(body.label)) throw new HttpError(400, "label is required, e.g. L");
    const runId = await queueBaseline(env, p, body.label, (body.name ?? `Builtin ${body.label}`).slice(0, 120));
    ctx.waitUntil(drive(env, runId).catch(() => undefined));
    return json({ run_id: runId }, 201);
  }],
];

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    try {
      for (const [method, re, handler] of routes) {
        const m = url.pathname.match(re);
        if (m && req.method === method) return await handler(req, env, ctx, m.slice(1));
      }
      if (url.pathname === "/") {
        return json({ service: "research-runs", docs: "see service/README.md", projects: "/v1/projects" });
      }
      throw new HttpError(404, "not found");
    } catch (err) {
      if (err instanceof HttpError) return json({ error: err.message }, err.status);
      console.error(err);
      return json({ error: "internal error", detail: err instanceof Error ? err.message : String(err) }, 500);
    }
  },

  async scheduled(_event: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(tick(env).then((r) => console.log("tick", JSON.stringify(r))));
  },
} satisfies ExportedHandler<Env>;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

async function readJson<T>(req: Request, optional = false): Promise<T> {
  const text = await req.text();
  if (!text) {
    if (optional) return {} as T;
    throw new HttpError(400, "expected a JSON body");
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new HttpError(400, "body is not valid JSON");
  }
}

function auth(req: Request, expected: string | undefined, name: string): void {
  if (!expected) throw new HttpError(503, `${name} is not configured on the server`);
  const got = req.headers.get("authorization")?.replace(/^Bearer\s+/i, "");
  if (got !== expected) throw new HttpError(401, "unauthorized");
}

async function mustProject(env: Env, slug: string): Promise<Project> {
  const p = await getProject(env, slug);
  if (!p) throw new HttpError(404, `no project ${slug}`);
  return p;
}

async function mustRun(env: Env, id: string): Promise<Run> {
  const r = await getRun(env, id);
  if (!r) throw new HttpError(404, `no run ${id}`);
  return r;
}

function publicProject(p: Project) {
  return {
    slug: p.slug,
    name: p.name,
    description: p.description,
    repo: p.repo,
    protocol: p.protocol,
    metric: p.metric_name,
    lower_is_better: !!p.lower_is_better,
    run_price_cents: p.run_price_cents,
    pool_cents: p.balance_cents,
  };
}

function publicRun(r: Run) {
  return {
    id: r.id,
    status: r.status,
    title: r.title,
    hypothesis: r.hypothesis,
    spec_source: r.spec_source,
    dispatcher: r.dispatcher,
    external_ref: r.external_ref,
    metric_value: r.metric_value,
    win_rate: r.win_rate,
    simulated: !!r.simulated,
    error: r.error,
    created_at: r.created_at,
    updated_at: r.updated_at,
  };
}
