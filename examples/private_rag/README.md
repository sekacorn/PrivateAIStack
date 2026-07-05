# Private RAG Example

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/knowledge/documents \
  -H "content-type: application/json" \
  -d '{"source_name":"example.md","content":"The service stores portable audit and memory records locally."}'

curl -sS -X POST http://127.0.0.1:8000/v1/knowledge/search \
  -H "content-type: application/json" \
  -d '{"query":"portable audit records","limit":3}'
```

Expected behavior: the search response cites a local source name and chunk ID. Inspect audit records with `make export-audit`.
