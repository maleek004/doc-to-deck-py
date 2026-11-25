# Retries for LLM calls

LLM APIs fail in two ways. Transient failures (HTTP 429, 500, 503, timeouts) usually succeed if you try again later. Permanent failures (HTTP 400, 401, a schema the model refuses) never will, so retrying them only wastes time and money.

## Exponential backoff

Wait longer after each failed attempt: 1 s, 2 s, 4 s, 8 s. Doubling the delay gives an overloaded server room to recover instead of hammering it at a fixed rate.

## Jitter

Add a random amount to each delay. Without jitter, every client that failed at the same moment retries at the same moment too, and the spikes repeat. This is the thundering herd problem.

## Retry-After

When a 429 or 503 response carries a `Retry-After` header, the server is telling you exactly how long to wait. Honour it, but cap it, so a buggy server cannot make you sleep for an hour.

```python
# never retry a cancellation: the caller asked us to stop
except asyncio.CancelledError:
    raise
```

# Structured output

Ask the model for JSON that matches a schema, then validate it anyway. The schema narrows what the model can say; validation catches what it still gets wrong, such as an index that points at a section that does not exist.
