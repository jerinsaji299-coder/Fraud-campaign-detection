# frontend

See the project README at the repository root — it is the complete,
authoritative document for this project, including the frontend's pages,
theme and colour system, API client, and tests.

Quick commands (from this directory):

```
npm install     # once
npm run dev     # dev server on http://localhost:5173
npm run build   # type-check + production bundle
npm run test    # Vitest
npm run lint    # oxlint
```

The API base URL comes from `VITE_API_URL` (see `.env.example`); it defaults
to `http://localhost:8000`.
