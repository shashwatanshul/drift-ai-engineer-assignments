# Caching

We put a Redis cache-aside layer in front of the product read path in March. The API
checks Redis first; on a miss it reads PostgreSQL, writes the row back with a 30-second
TTL, and returns it. Writes from the admin tool delete the key rather than updating it,
so the next reader re-populates from the source of truth.

Hit rate settled at 88% after the first week, which took PostgreSQL read load from roughly
10k queries per second down to 1.2k. The p99 on cache hits is under 2ms against 180ms for
the database path.

The one incident so far was a stampede: a Redis failover emptied the cache during a
campaign and every pod hit PostgreSQL at once. We now hold a short single-flight lock per
key, so only the first request for a cold key goes to the database while the rest wait.
