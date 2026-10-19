## Group observations into time buckets
Grafana time series can aggregate observations into buckets. In a PostgreSQL
query, the `time` alias identifies the time column and `metric` separates series.
This example uses fictional events and categories.

```sql
SELECT
    $__timeGroup(occurred_at, '14d') AS time,
    category AS metric,
    COUNT(*) AS event_count
FROM example_event
GROUP BY 1, 2;
```

Each time bucket receives one count per category. Choose bucket size and the
aggregation to match the question the graph should answer.

---
Associations: [[grafana]], [[SQL]]
