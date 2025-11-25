# Emergency Department Frontend

React + TypeScript single-page app that consumes the Flask backend in `../backend`. The UI surfaces operational KPIs, live encounter feeds, model-driven insights, and tooling for prediction endpoints.

## Tech stack

- [Vite](https://vite.dev/) for bundling and dev server
- React 18 with React Router for client-side routing
- [@tanstack/react-query](https://tanstack.com/query) for API data management and caching
- [Recharts](https://recharts.org/en-US/) for visualizations
- Axios-powered API layer with support for configurable base URLs (`VITE_API_BASE_URL`)

## Getting started

```bash
cd frontend
cp .env.example .env                # optional: point to remote backend
npm install
npm run dev                        # http://localhost:5173
```

The frontend expects the Flask API to be running locally on port 5000 (or the URL you set in `.env`).

### Available scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start Vite dev server with HMR |
| `npm run build` | Create production build in `dist/` |
| `npm run preview` | Preview the production build |
| `npm run lint` | Run ESLint (from Vite template) |

## Application structure

```
frontend/
├── src/
│   ├── App.tsx              # Shell layout + routing
│   ├── main.tsx             # React/Router/Query bootstrapping
│   ├── lib/api.ts           # Axios client + typed helpers
│   ├── types/api.ts         # Backend response/DTO types
│   ├── pages/
│   │   ├── Dashboard.tsx    # KPI cards, charts, vitals, diagnoses
│   │   ├── Encounters.tsx   # Filterable encounter feed
│   │   ├── Predictions.tsx  # Forms for ESI, wait time, volume models
│   │   └── Staff.tsx        # Staff roster + role breakdown
│   └── App.css + index.css  # Tailored design system
└── .env.example             # Base URL configuration
```

## Feature highlights

- **Dashboard** pulls `/statistics/*` endpoints for totals, ESI distributions, arrival modes, vitals, payor mix, LWBS risk, and top diagnoses (visualized with Recharts).
- **Encounters** page streams `/api/encounters` with query-string filters for ESI level and pagination.
- **Predictions** page displays `/api/predictions/models/info` metadata and provides three forms that call `/predictions/esi`, `/predictions/wait-time`, and `/predictions/volume`.
- **Staff** page fetches `/api/staff` with toggle-able `active_only` flag and aggregates counts per role code.

## Environment variables

- `VITE_API_BASE_URL` – default `http://localhost:5000/api`. Set this when pointing to a remote instance.

## Next steps / ideas

- Integrate auth (if required) and user management.
- Persist user-defined filters in URL query params.
- Add charts for encounter trends and wait-time history once backend endpoints land.
- Layer in form validation + presets from the dataset catalog.
