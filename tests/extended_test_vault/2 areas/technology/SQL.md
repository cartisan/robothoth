## Lateral joins
A lateral subquery can refer to columns from an earlier table in the query.
`LEFT JOIN LATERAL` keeps the outer row even if the subquery finds no match.
This example uses fictional items and tags, not a production schema.

```sql
SELECT item.id, tags.tag_count
FROM example_item AS item
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS tag_count
    FROM jsonb_array_elements_text(item.tags) AS tag
) AS tags ON TRUE;
```

`jsonb_array_elements_text` expands a JSON array into rows. The subquery counts
those rows for each item; `ON TRUE` introduces no further join condition.

## Common table expressions
A CTE names a temporary query result for use within the same statement.
It can make intermediate steps easier to read and reuse.

```sql
WITH category_totals AS (
    SELECT category, COUNT(*) AS item_count
    FROM example_item
    GROUP BY category
)
SELECT category, item_count
FROM category_totals
WHERE item_count > 1;
```

---
Associations: [[database]]
