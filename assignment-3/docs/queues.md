# Queues

Background work runs through RabbitMQ with three queues: `emails` (transactional mail),
`exports` (long-running CSV generation), and `webhooks` (outbound delivery to customers).
Each has its own consumer pool so a backlog in one cannot starve the others.

Delivery is at-least-once, so every consumer must be idempotent. We key on a `job_id` in
the message body and record completed IDs for 24 hours, which is long enough to absorb a
redelivery after a consumer crash.

Failed jobs retry three times with exponential backoff, then land in a per-queue dead
letter queue. Nothing drains the DLQs automatically — someone on call looks at them each
morning. `exports` is the usual occupant, normally because a customer requested a date
range wide enough to time out.
