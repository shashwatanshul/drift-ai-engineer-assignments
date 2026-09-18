# Product API — constraints

- **Staleness:** product data may be up to 30 seconds stale. Price changes are the
  sensitive field; merchandising accepts 30s but not minutes.
- **Correctness on write:** after the admin tool saves, the next read from that admin
  session must show the new value. Eventual consistency is fine for everyone else.
- **Budget:** no new managed services this quarter. Reusing the existing Redis cluster
  is approved; standing up a new CDN contract is not.
- **Traffic shape:** strongly skewed. A small set of products drives most of the reads,
  and the skew gets worse during campaigns.
- **Failure policy:** a cache outage must degrade to direct database reads, not to
  errors. The database cannot take the full unbuffered read load at peak, so any design
  needs an answer for the stampede that follows a cold cache.
