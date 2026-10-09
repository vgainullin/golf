-- Real payments: a processor may deliver the same payment more than once, so
-- each payment reference can fund the pool only once.
CREATE UNIQUE INDEX donations_payment_ref ON donations(payment_ref);
