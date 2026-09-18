# Observability

Metrics go to Prometheus and dashboards live in Grafana. Every service exposes the four
golden signals — request rate, error rate, latency percentiles, and saturation — under a
common set of label names, so one dashboard template works across services.

Logs are structured JSON shipped to Loki. Every request carries a trace ID generated at
the edge and threaded through downstream calls, which is what makes a log search and a
trace line up. Traces themselves go to Tempo with 10% head sampling, raised to 100% for
requests that end in an error.

Alerting is deliberately thin: we page only on symptoms customers feel — elevated 5xx,
p99 latency past the SLO, or queue depth growing without a matching drain rate. Cause
alerts such as high CPU go to a Slack channel rather than to a pager.
