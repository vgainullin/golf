#!/usr/bin/env node
// CLI for the research-runs API. No dependencies; needs Node 18+.
//
//   research projects
//   research project golf
//   research donate golf --amount 20 [--donor NAME] [--note TEXT]
//                        [--code agent.py | --checkpoint URL] [--title T] [--hypothesis H]
//   research leaderboard golf [--protocol ID]
//   research runs golf
//   research run RUN_ID [--watch]
//   research report RUN_ID
//   research tick                                   (ADMIN_TOKEN)
//   research runner [--once] [--python PATH] [--games-per-perm N]   (RUNNER_TOKEN)
//
// Env: RESEARCH_API (default http://localhost:8787), RUNNER_TOKEN, ADMIN_TOKEN.
// Add --json to any read command for the raw response.
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const API = (process.env.RESEARCH_API ?? "http://localhost:8787").replace(/\/$/, "");
const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..", "..");

function parseArgs(argv) {
  const pos = [];
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) flags[key] = true;
      else flags[key] = argv[++i];
    } else pos.push(a);
  }
  return { pos, flags };
}

async function call(method, p, { body, token } = {}) {
  const res = await fetch(API + p, {
    method,
    headers: {
      ...(body !== undefined ? { "content-type": "application/json" } : {}),
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 204) return null;
  const text = await res.text();
  const data = res.headers.get("content-type")?.includes("json") ? JSON.parse(text) : text;
  if (!res.ok) {
    const msg = typeof data === "object" ? data.error ?? JSON.stringify(data) : data;
    throw new Error(`${method} ${p}: HTTP ${res.status}: ${msg}`);
  }
  return data;
}

const usd = (c) => `$${(c / 100).toFixed(2)}`;
const fmt = (v, d = 3) => (v == null ? "-" : Number(v).toFixed(d));
const pct = (v) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);

function table(rows, cols) {
  const widths = cols.map(([h, f]) => Math.max(h.length, ...rows.map((r) => String(f(r)).length)));
  const line = (cells) => cells.map((c, i) => String(c).padEnd(widths[i])).join("  ");
  console.log(line(cols.map(([h]) => h)));
  for (const r of rows) console.log(line(cols.map(([, f]) => f(r))));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const commands = {
  async projects({ flags }) {
    const { projects } = await call("GET", "/v1/projects");
    if (flags.json) return console.log(JSON.stringify(projects, null, 2));
    table(projects, [
      ["slug", (p) => p.slug],
      ["name", (p) => p.name],
      ["run price", (p) => usd(p.run_price_cents)],
      ["pool", (p) => usd(p.pool_cents)],
      ["protocol", (p) => p.protocol],
    ]);
  },

  async project({ pos, flags }) {
    const d = await call("GET", `/v1/projects/${pos[0]}`);
    if (flags.json) return console.log(JSON.stringify(d, null, 2));
    const p = d.project;
    console.log(`${p.name} (${p.slug})  repo ${p.repo}`);
    console.log(p.description);
    console.log(`\nOne run costs ${usd(p.run_price_cents)}. Pool: ${usd(p.pool_cents)}.`);
    console.log(`Scored on ${p.protocol}: ${p.metric}, ${p.lower_is_better ? "lower" : "higher"} is better.`);
    if (d.best) console.log(`Best: ${d.best.name} at ${fmt(d.best.metric_value)}${d.best.simulated ? " (simulated)" : ""}`);
    const s = Object.entries(d.runs_by_status).map(([k, v]) => `${k} ${v}`).join(", ");
    console.log(`Runs: ${s || "none yet"}`);
  },

  async donate({ pos, flags }) {
    const slug = pos[0];
    if (!slug || !flags.amount) throw new Error("usage: research donate SLUG --amount USD [--donor NAME] [--note TEXT] [--code FILE | --checkpoint URL]");
    const body = { donor: flags.donor ?? process.env.USER ?? "anonymous", amount_usd: Number(flags.amount) };
    if (typeof flags.note === "string") body.note = flags.note;
    if (flags.code || flags.checkpoint) {
      body.submission = {
        title: flags.title,
        hypothesis: flags.hypothesis,
        spec: flags.code
          ? { kind: "agent_code", agent_code: readFileSync(flags.code, "utf8") }
          : { kind: "checkpoint", checkpoint: { url: flags.checkpoint } },
      };
    }
    const d = await call("POST", `/v1/projects/${slug}/donations`, { body });
    if (flags.json) return console.log(JSON.stringify(d, null, 2));
    console.log(`Donation ${d.donation_id} recorded (payment stubbed). ${d.message}`);
    for (const id of d.runs) console.log(`  run ${id}   follow with: research run ${id} --watch`);
  },

  async leaderboard({ pos, flags }) {
    const q = typeof flags.protocol === "string" ? `?protocol=${encodeURIComponent(flags.protocol)}` : "";
    const d = await call("GET", `/v1/projects/${pos[0]}/leaderboard${q}`);
    if (flags.json) return console.log(JSON.stringify(d, null, 2));
    console.log(`${d.project} on ${d.protocol}: ${d.metric} (${d.lower_is_better ? "lower" : "higher"} is better)\n`);
    table(d.rows, [
      ["#", (r) => r.rank],
      ["name", (r) => r.name + (r.simulated ? " [sim]" : "")],
      [d.metric, (r) => fmt(r.metric_value)],
      ["win", (r) => pct(r.win_rate)],
      ["source", (r) => r.source],
      ["run", (r) => r.run_id ?? ""],
    ]);
    const others = d.protocols.filter((p) => p !== d.protocol);
    if (others.length) console.log(`\nOther boards: ${others.join(", ")} (use --protocol)`);
  },

  async runs({ pos, flags }) {
    const { runs } = await call("GET", `/v1/projects/${pos[0]}/runs`);
    if (flags.json) return console.log(JSON.stringify(runs, null, 2));
    table(runs, [
      ["id", (r) => r.id],
      ["status", (r) => r.status],
      ["score", (r) => fmt(r.metric_value) + (r.simulated ? " [sim]" : "")],
      ["by", (r) => r.spec_source],
      ["title", (r) => r.title ?? ""],
    ]);
  },

  async run({ pos, flags }) {
    let last = "";
    for (;;) {
      const d = await call("GET", `/v1/runs/${pos[0]}`);
      if (flags.json && !flags.watch) return console.log(JSON.stringify(d, null, 2));
      const r = d.run;
      const line = `${r.id}  ${r.status}  ${r.title ?? ""}  ${r.metric_value != null ? fmt(r.metric_value) : ""}`;
      if (!flags.watch) {
        console.log(line);
        if (r.hypothesis) console.log(`hypothesis: ${r.hypothesis}`);
        if (r.external_ref) console.log(`executor: ${r.external_ref}`);
        if (r.error) console.log(`error: ${r.error}`);
        console.log("\nevents:");
        for (const e of d.events) console.log(`  ${e.at}  ${e.type.padEnd(8)} ${e.message}`);
        if (r.report_md) console.log(`\nreport ready: research report ${r.id}`);
        return;
      }
      if (line !== last) console.log(`${new Date().toISOString().slice(11, 19)}  ${line}`);
      last = line;
      if (r.status === "reported" || r.status === "failed") {
        if (r.report_md) console.log("\n" + r.report_md);
        return;
      }
      await sleep(3000);
    }
  },

  async report({ pos }) {
    process.stdout.write(await call("GET", `/v1/runs/${pos[0]}/report`));
  },

  async tick({ flags }) {
    const d = await call("POST", "/v1/admin/tick", { token: process.env.ADMIN_TOKEN });
    if (flags.json) return console.log(JSON.stringify(d, null, 2));
    console.log(`considered ${d.considered} run(s)`);
    for (const m of d.moved) console.log(`  ${m.id} -> ${m.status}`);
  },

  // Pull-based runner for DISPATCHER=queue: claim a run, execute it with
  // service/runner/execute.py in this checkout, post the result.
  async runner({ flags }) {
    const token = process.env.RUNNER_TOKEN;
    if (!token) throw new Error("set RUNNER_TOKEN");
    const python = flags.python ?? path.join(repoRoot, ".venv", "bin", "python");
    const name = flags.name ?? `cli-${process.pid}`;
    for (;;) {
      const job = await call("POST", "/v1/runner/claim", { token, body: { runner: name } });
      if (!job) {
        if (flags.once) return console.log("no queued runs");
        await sleep(5000);
        continue;
      }
      console.log(`claimed ${job.run_id} (${job.spec.kind})`);
      const dir = mkdtempSync(path.join(tmpdir(), "research-run-"));
      const specPath = path.join(dir, "spec.json");
      const outPath = path.join(dir, "result.json");
      writeFileSync(specPath, JSON.stringify(job.spec));
      const args = ["-m", "service.runner.execute", "--spec", specPath, "--out", outPath];
      if (flags["games-per-perm"]) args.push("--games-per-perm", String(flags["games-per-perm"]));
      const code = await new Promise((resolve) => {
        const child = spawn(python, args, { cwd: repoRoot, stdio: "inherit", env: { ...process.env, RUNNER_TOKEN: "" } });
        child.on("exit", resolve);
        child.on("error", (e) => {
          console.error(e.message);
          resolve(-1);
        });
      });
      let result;
      try {
        result = JSON.parse(readFileSync(outPath, "utf8"));
      } catch {
        result = { status: "error", error: `executor exited ${code} without writing a result` };
      }
      const d = await call("POST", `/v1/runner/runs/${job.run_id}/result`, { token, body: result });
      console.log(`posted result for ${job.run_id}: ${d.run.status} ${fmt(d.run.metric_value)}`);
      if (flags.once) return;
    }
  },
};

const [cmd, ...rest] = process.argv.slice(2);
if (!cmd || !commands[cmd]) {
  console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").slice(1, 15).map((l) => l.replace(/^\/\/ ?/, "")).join("\n"));
  process.exit(cmd ? 1 : 0);
}
commands[cmd](parseArgs(rest)).catch((e) => {
  console.error(e.message);
  process.exit(1);
});
