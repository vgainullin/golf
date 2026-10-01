/**
 * Loads a project's experiment journal: the write-ups of past experiments that
 * live in its repo (for golf, docs/experiments.md and friends). The planner
 * reads it before deciding what to try.
 */
import type { Env, Project } from "./types";

export interface JournalFile {
  path: string;
  text: string;
}

const MAX_TOTAL_CHARS = 400_000;

export async function loadJournal(env: Env, project: Project): Promise<JournalFile[]> {
  const paths = project.journal_paths
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
  const files: JournalFile[] = [];
  let total = 0;
  for (const path of paths) {
    const text = await fetchRepoFile(env, project.repo, path);
    if (text == null) continue;
    const room = MAX_TOTAL_CHARS - total;
    if (room <= 0) break;
    files.push({ path, text: text.length > room ? text.slice(0, room) + "\n[truncated]" : text });
    total += Math.min(text.length, room);
  }
  return files;
}

async function fetchRepoFile(env: Env, repo: string, path: string): Promise<string | null> {
  // Contents API works for private repos when a token is set; raw works for public ones.
  const url = env.GITHUB_TOKEN
    ? `https://api.github.com/repos/${repo}/contents/${path}?ref=${encodeURIComponent(env.GITHUB_REF)}`
    : `https://raw.githubusercontent.com/${repo}/${env.GITHUB_REF}/${path}`;
  const headers: Record<string, string> = { "User-Agent": "research-runs" };
  if (env.GITHUB_TOKEN) {
    headers.Authorization = `Bearer ${env.GITHUB_TOKEN}`;
    headers.Accept = "application/vnd.github.raw+json";
  }
  try {
    const res = await fetch(url, { headers });
    if (!res.ok) {
      console.warn(`journal: ${path} HTTP ${res.status}`);
      return null;
    }
    return await res.text();
  } catch (err) {
    console.warn(`journal: ${path} ${err}`);
    return null;
  }
}
