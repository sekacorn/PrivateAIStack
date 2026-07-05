# Private Code Review Example

Review the sample target:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/reviews \
  -H "content-type: application/json" \
  -d '{"repository_path":"sample-target","mode":"safe-static"}'
```

Fetch the Markdown report:

```bash
curl -sS "http://127.0.0.1:8000/v1/reviews/REVIEW_ID/report?format=markdown"
```

The sample source should remain unchanged.
