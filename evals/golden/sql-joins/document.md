# SQL joins

A join combines rows from two tables by matching a column in one against a column in the other. The examples use two small tables.

`customers`:

| id | name  |
|----|-------|
| 1  | Ada   |
| 2  | Grace |
| 3  | Linus |

`orders`:

| id  | customer_id | total |
|-----|-------------|-------|
| 10  | 1           | 25.00 |
| 11  | 1           | 40.00 |
| 12  | 2           | 15.00 |
| 13  | 9           | 60.00 |

Linus has no orders, and order 13 belongs to a customer id that does not exist.

## INNER JOIN

An inner join returns only the rows that have a match in both tables.

```sql
SELECT c.name, o.total
FROM customers AS c
INNER JOIN orders AS o ON o.customer_id = c.id;
```

| name  | total |
|-------|-------|
| Ada   | 25.00 |
| Ada   | 40.00 |
| Grace | 15.00 |

Linus and order 13 are both missing from the result: neither has a partner.

## LEFT JOIN

A left join (LEFT OUTER JOIN) keeps every row of the left table. Where the right table has no match, its columns are filled with NULL.

```sql
SELECT c.name, o.total
FROM customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.id;
```

| name  | total |
|-------|-------|
| Ada   | 25.00 |
| Ada   | 40.00 |
| Grace | 15.00 |
| Linus | NULL  |

## Finding rows with no match

Adding `WHERE o.id IS NULL` to a left join keeps only the left rows that found no partner. This pattern is called an anti-join; here it finds the customers who never ordered.

```sql
SELECT c.name
FROM customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.id
WHERE o.id IS NULL;   -- returns: Linus
```

## FULL OUTER JOIN and CROSS JOIN

A full outer join keeps the unmatched rows from both sides, with NULLs on whichever side is missing; here it returns Linus and order 13 as well as the matches. MySQL does not support FULL OUTER JOIN; it is emulated with a UNION of a left join and a right join.

A cross join pairs every row of one table with every row of the other, with no condition: the Cartesian product. Three customers and four orders give 3 × 4 = 12 rows.

## A common mistake

A condition on the right table placed in the WHERE clause silently turns a left join into an inner join, because NULL never satisfies a comparison such as `o.total > 20`. To filter the right table and still keep every left row, put the condition in the ON clause instead.

## Running the examples

```python
import sqlite3

# sqlite3 is in the standard library: no server to install
db = sqlite3.connect(":memory:")
# Load the two tables, then run any query above
```

The sample data is fictional and was made up for these notes.
