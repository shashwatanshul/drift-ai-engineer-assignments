# Product API — architecture notes

The read path is a single `GET /v1/products/{id}` endpoint backed by PostgreSQL.
Six stateless API pods sit behind an L7 load balancer; there is no cache layer today,
so every read is a database query.

Reads dominate: catalogue browsing and the mobile home screen both fan out to this
endpoint. Writes come only from the merchandising admin tool and a nightly importer.

Responses are JSON product documents assembled from four tables (product, pricing,
inventory summary, media URLs). Assembly is roughly 60% of the p99 latency; the raw
queries themselves are fast.

We already run a Redis cluster for session storage. It has spare capacity and the team
is on call for it, so reusing it is cheaper operationally than adding new infrastructure.
