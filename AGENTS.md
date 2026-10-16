# AGENTS.md

Rules for every coding agent working in this repository.

## Required: Ponytail

Ponytail is mandatory for every agent and every session in this repo. It enforces the smallest complete change while never cutting validation, error handling, security, or tests for risky logic.

Install from the official sources only: `DietrichGebert/ponytail` on GitHub or `@dietrichgebert/ponytail` on npm.

| Agent | Install |
|---|---|
| Claude Code (local) | Preconfigured in `.claude/settings.json`; accept the prompt when you trust the folder. Manual: `/plugin marketplace add DietrichGebert/ponytail` then `/plugin install ponytail@ponytail` |
| Claude cloud sessions (web, mobile, `--cloud`) | Nothing to install: the skills are committed in `.claude/skills/ponytail*` (upstream `84740e3`, MIT) and load automatically. Level is `full` |
| Codex | `codex plugin marketplace add DietrichGebert/ponytail` then `codex plugin add ponytail@ponytail` |
| Anything else | Read `.claude/skills/ponytail/SKILL.md` before starting |

Use:
- Default mode `full`.
- Run `/ponytail-review` on every diff before declaring done. Fix all "must fix" findings.
- Run `/ponytail-audit` after any large refactor, `/ponytail-debt` before a release.
- Shortcut format: `shortcut: <the limit>, <when to upgrade>`.
- CI: the `ponytail` workflow reviews every non-draft PR and fails on any "must fix". It needs the `OPENROUTER_API_KEY` repo secret; without it the check skips with a warning.

If Ponytail cannot be loaded in the current environment, say so at the top of your reply and apply the rules in `.claude/skills/ponytail/SKILL.md` by hand.
