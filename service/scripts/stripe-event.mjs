// Prints a curl command that posts a signed, paid checkout.session.completed
// event to the local Worker, the way Stripe would. Used by e2e.sh.
//   node scripts/stripe-event.mjs SESSION_ID AMOUNT_CENTS [PROJECT]
import { createHmac } from "node:crypto";

const [id, cents, project = "golf"] = process.argv.slice(2);
const api = process.env.RESEARCH_API ?? "http://localhost:8787";
const secret = process.env.STRIPE_WEBHOOK_SECRET ?? "whsec_dev";
const body = JSON.stringify({
  type: "checkout.session.completed",
  data: { object: { id, amount_total: Number(cents), currency: "usd", payment_status: "paid", client_reference_id: project } },
});
const t = Math.floor(Date.now() / 1000);
const sig = createHmac("sha256", secret).update(`${t}.${body}`).digest("hex");
const q = (s) => `'${s.replace(/'/g, `'\\''`)}'`;
console.log(`curl -sS -X POST ${q(`${api}/v1/webhooks/stripe`)} -H ${q(`stripe-signature: t=${t},v1=${sig}`)} -H 'content-type: application/json' --data-binary ${q(body)}`);
