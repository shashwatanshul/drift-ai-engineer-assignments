# Indexing

Product search runs on OpenSearch. A change-data-capture stream tails the PostgreSQL WAL
and pushes updates into the index, so a product edit is searchable within about four
seconds. There is also a nightly full reindex that rebuilds into a new index and swaps an
alias over on completion, which gives us a clean rollback if a mapping change goes wrong.

The index holds roughly 1.8 million documents across six shards. Analysers are configured
per language field; we run English and German today and adding a third is mostly a
mapping change plus a reindex.

The known weak spot is that CDC lag is invisible to the API. If the stream stalls, search
quietly serves stale results and nothing alerts. Wiring lag into the health check is on
the backlog.
