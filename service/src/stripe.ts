/**
 * Stripe Checkout webhook: a paid Checkout Session becomes a donation.
 *
 * The site's donate buttons open a Stripe Payment Link with
 * client_reference_id=<project slug>. Stripe calls POST /v1/webhooks/stripe
 * with checkout.session.completed (or async_payment_succeeded for delayed
 * methods); we verify the signature and credit the project's pool once per
 * session id.
 */
import { getProject } from "./db";
import { donate, HttpError } from "./pipeline";
import type { Env } from "./types";

const TOLERANCE_SECONDS = 300;

interface CheckoutSession {
  id: string;
  amount_total: number | null;
  currency: string | null;
  payment_status: string;
  client_reference_id: string | null;
  metadata?: Record<string, string> | null;
}

export async function handleStripeWebhook(req: Request, env: Env): Promise<{ status: string; runs: string[] }> {
  if (!env.STRIPE_WEBHOOK_SECRET) throw new HttpError(503, "STRIPE_WEBHOOK_SECRET is not configured on the server");
  const body = await req.text();
  await verifySignature(body, req.headers.get("stripe-signature"), env.STRIPE_WEBHOOK_SECRET);

  const event = JSON.parse(body) as { type: string; data: { object: CheckoutSession } };
  if (event.type !== "checkout.session.completed" && event.type !== "checkout.session.async_payment_succeeded") {
    return { status: `ignored ${event.type}`, runs: [] };
  }
  const session = event.data.object;
  if (session.payment_status !== "paid") return { status: `not paid yet (${session.payment_status})`, runs: [] };
  if ((session.currency ?? "").toLowerCase() !== "usd" || !session.amount_total) {
    // Acknowledge so Stripe stops retrying, but leave a trace to reconcile by hand.
    console.error("stripe: unsupported payment", session.id, session.currency, session.amount_total);
    return { status: "ignored: only USD payments are pooled", runs: [] };
  }

  const wanted = session.metadata?.project ?? session.client_reference_id ?? "";
  const project = (await getProject(env, wanted)) ?? (await getProject(env, env.DEFAULT_PROJECT || "golf"));
  if (!project) throw new HttpError(500, "no project to credit");

  const out = await donate(env, project, "anonymous", session.amount_total, null, null, `stripe:${session.id}`);
  return { status: out.duplicate ? "duplicate" : `credited ${project.slug}`, runs: out.runs };
}

/** Stripe-Signature: t=<unix>,v1=<hex hmac-sha256 of "t.body">[,v1=...] */
async function verifySignature(body: string, header: string | null, secret: string): Promise<void> {
  if (!header) throw new HttpError(400, "missing Stripe-Signature");
  const parts = header.split(",").map((p) => p.split("=", 2) as [string, string]);
  const t = parts.find(([k]) => k === "t")?.[1];
  const sigs = parts.filter(([k]) => k === "v1").map(([, v]) => v);
  if (!t || !sigs.length) throw new HttpError(400, "malformed Stripe-Signature");
  if (Math.abs(Date.now() / 1000 - Number(t)) > TOLERANCE_SECONDS) throw new HttpError(400, "stale Stripe-Signature");

  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, [
    "sign",
  ]);
  const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(`${t}.${body}`)));
  const expected = [...mac].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (!sigs.some((s) => timingSafeEqual(s, expected))) throw new HttpError(400, "bad Stripe-Signature");
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
