# frontend/ — Next.js Dashboard

Web UI for dataset upload, pipeline configuration, live training monitoring, and results browsing.

Back to [project root](../README.md)

---

## Setup

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

Requires the [API server](../api/README.md) running at `http://localhost:8000`.

---

## Pages

| Route | File | Description |
|-------|------|-------------|
| `/` | `src/app/page.tsx` | Dashboard — status overview |
| `/datasets` | `src/app/datasets/page.tsx` | Upload and manage datasets |
| `/pipeline` | `src/app/pipeline/page.tsx` | Configure and start a training run |
| `/pipeline/[runId]` | `src/app/pipeline/[runId]/page.tsx` | Live training progress + results |
| `/results` | `src/app/results/page.tsx` | Browse all generated image sets |
| `/docs` | `src/app/docs/page.tsx` | Patent disclosure + API reference |

---

## Workflow via UI

1. **Upload dataset** at `/datasets` — zip file with `class_name/image.jpg` layout inside
2. **Configure run** at `/pipeline` — select dataset, set epochs, architecture, image size
3. **Watch training** at `/pipeline/[runId]` — live loss curves via WebSocket
4. **View results** — synthetic image grids and quality metrics on the same page

---

## Key Source Files

```
src/
├── lib/
│   ├── api.ts               # typed fetch client — matches all FastAPI endpoints
│   ├── types.ts             # TypeScript types for API responses
│   └── use-training-ws.ts   # WebSocket hook — connects to /ws/training/{runId}
└── components/
    ├── image-grid.tsx        # paginated image display
    ├── quality-metrics.tsx   # metric cards (FID, SSIM, PSNR, diversity)
    ├── training-chart.tsx    # loss curve chart (recharts)
    ├── sidebar.tsx           # navigation sidebar
    └── equation-block.tsx    # LaTeX rendering for docs page
```

---

## API Base URL

Default: `http://localhost:8000`

To change, set `NEXT_PUBLIC_API_URL` in `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://your-api-host:8000
```

---

## Tech Stack

- Next.js 16 (App Router)
- React 19
- TypeScript
- Tailwind CSS
