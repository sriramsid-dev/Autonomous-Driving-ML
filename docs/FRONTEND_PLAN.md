# Frontend plan — React + Vite over FastAPI over the real pipeline

Status: **plan only, nothing implemented.** Written after the lane evaluation
reached a defensible result (`EVALUATION.md` §15), because the honest UI is
shaped by those numbers: the pipeline runs at **~3 FPS on CPU**, the lane model
choice changes both accuracy and coverage, and traffic-light states include an
explicit `unknown`. A UI that implies a live road-ready system would be a lie.

## What the backend wraps

The pipeline already exists as a class, so the API is a thin adapter, not a
rewrite:

```python
from main import PerceptionPipeline
pipe = PerceptionPipeline(enable_objects=True,
                          enable_traffic_lights=True,
                          enable_lanes=True,
                          device="cpu")
annotated_bgr, telemetry = pipe.process_frame(frame_bgr, frame_idx, total)
```

`telemetry` already carries everything the UI needs:

```jsonc
{
  "stats": { "frame_idx", "total_frames", "fps", "latency_ms",
             "vehicles_count", "pedestrians_count", "active_tracks",
             "tl_state", "lane_status" },
  "tracked_objects": [ { "bbox", "track_id", "class", "conf", ... } ],
  "tl_detections":  [ { "bbox", "state", "confidence", ... } ],
  "lanes":          [ [ [x, y], ... ] ]        // per lane slot
}
```

### One small change the plan depends on

`PerceptionPipeline.__init__` constructs `UltraFastLaneDetector(device=device)`
with no `model_key` / `preprocess`, so today the lane profile can only be chosen
by calling the modules directly or via environment variables before import.
The plan adds optional keyword arguments (`lane_model="tusimple"`,
`lane_preprocess="stretch"`) that are forwarded to the detector. The CLI keeps
working unchanged; this is additive.

## Architecture

```
React (Vite, TS)  ──REST──▶  FastAPI  ──in-process──▶  PerceptionPipeline
      │                        │                         ├─ Module A YOLO+ByteTrack
      │──SSE (frames+meta)────▶│                         ├─ Module B detector+classifier
      └────────────────────────┴─ static: outputs/runs/  └─ Module C UFLD
```

Single process, one pipeline instance built at startup so the three model loads
(~1-2 s) happen once. One worker thread; a `threading.Lock` guards the pipeline
because ByteTrack and the lane temporal filter are stateful and must not be
entered concurrently. Multi-worker uvicorn is explicitly **not** used.

## API surface

| method | path | purpose |
|---|---|---|
| GET | `/api/health` | readiness, device, loaded modules, model file presence |
| GET | `/api/config` | lane profiles (`tusimple`, `culane`), preprocess options, module flags |
| POST | `/api/infer/image` | multipart image → `{annotated_jpeg_b64, telemetry, timing}` |
| POST | `/api/infer/image?stream=false` | same but returns the annotated file URL instead of base64, for large frames |
| POST | `/api/jobs` | start a video job: `{source: "upload" \| "clip:<id>", max_frames, lane_model, modules}` → `{job_id}` |
| GET | `/api/jobs/{id}/events` | **SSE**: `frame` events carrying `frameIdx`, `jpeg` (or `jpegRef`), `stats`, `lanes`, `tl` |
| GET | `/api/jobs/{id}/telemetry` | full JSONL of per-frame telemetry after completion |
| POST | `/api/jobs/{id}/cancel` | cooperative stop; sets a flag the loop checks |
| GET | `/api/clips` | the bundled evaluation clips with licence + attribution from `sources.json` |
| GET | `/api/eval/lane` | the committed lane `report.json` for the four configurations |
| GET | `/api/eval/tl` | the committed traffic-light `report.json` + confusion matrix |
| GET | `/api/media/{path}` | static, path-validated access to `outputs/runs/**` |

Design notes that follow from the measurements:

* **No websocket video.** At 2.8-3 FPS the honest presentation is one JPEG per
  analysed frame over SSE, with the loop skipping source frames rather than
  pretending real-time. The UI shows the measured `latency_ms` and a
  "3.0 FPS · CPU" badge next to the video, not a spinner.
* **`lane_model` is a first-class request option**, because §15 shows the two
  profiles trade coverage (6/13 vs 3/13 frames) against accuracy (17 px vs
  7.4 px median). The UI must show both numbers when the toggle is flipped, not
  just the prettier one.
* **`unknown` is rendered as its own state** (grey, hatched), never coerced to
  green/red. The classifier abstains deliberately.
* Jobs run in a background thread writing frames to
  `outputs/runs/api_<timestamp>/`, so an interrupted job leaves an inspectable
  directory rather than a partial response.

## Frontend structure

```
frontend/
  index.html
  vite.config.ts            # proxy /api -> http://127.0.0.1:8000
  src/
    main.tsx
    App.tsx                 # routes: Capture | Analyse | Evaluation | Limits
    api/client.ts           # fetch wrappers + zod schemas for telemetry
    api/sse.ts              # EventSource hook with reconnect + cancel
    routes/Capture.tsx      # single image, module toggles, lane model
    routes/Analyse.tsx      # video job: clip picker or upload, progress, frame canvas
    routes/Evaluation.tsx   # lane + traffic-light tables read from /api/eval/*
    routes/Limits.tsx       # the limitations, copied from EVALUATION.md
    components/
      FrameCanvas.tsx       # <canvas> JPEG + optional SVG overlay from telemetry
      TelemetryPanel.tsx    # fps, latency, counts, tl_state, lane_status
      LaneModelToggle.tsx   # tusimple|culane + the coverage/accuracy hint
      ModuleToggles.tsx     # A / B / C
      RunBadge.tsx          # "experimental · not for vehicle control"
      MetricTable.tsx       # generic table fed by the eval JSON
```

* **Stack:** Vite + React 18 + TypeScript, TanStack Query for REST, plain
  `EventSource` for SSE, no component library beyond CSS modules — the project
  has no UI dependencies today and the plan does not add a design system.
* **Overlay:** the canvas draws the annotated JPEG the backend already produces
  (`annotate_frame`), so the browser never re-implements rendering. An optional
  SVG layer draws `tracked_objects` boxes for hover/tooltip, driven by the same
  telemetry, so id-switches are visible when scrubbing a recorded run.
* **Evaluation route is read-only.** It renders the committed
  `report.json` files; it never recomputes metrics in the browser, and it labels
  "measured" vs "not measured" explicitly, mirroring `EVALUATION.md`.

## Local run

```powershell
python -m venv .venv; .\.venv\Scripts\activate
pip install -r requirements-api.txt          # fastapi, uvicorn, python-multipart
uvicorn api.app:app --host 127.0.0.1 --port 8000
cd frontend; npm install; npm run dev        # http://127.0.0.1:5173
```

Bound to loopback only; no auth, because this is a local single-user analysis
tool. Uploads are capped (e.g. 50 MB) and written to a per-job temp directory
that is deleted on cancel/complete unless `--keep`.

## Tests the plan requires before it is called done

1. `pytest api/` — `TestClient`: health, image inference returns a decodable
   JPEG and a schema-valid telemetry object, a job streams ≥1 frame event and
   completes, cancel stops early, path traversal on `/api/media` returns 404.
2. One Playwright smoke test: capture page uploads a bundled image, canvas
   paints, `lane_model` toggle issues a request with the new value.
3. A regression guard that the API imports `PerceptionPipeline` rather than
   shelling out to `main.py`, so CLI behaviour cannot drift from API behaviour.

## Non-goals

* No vehicle control, no actuation, no real-time guarantee — the system is an
  experimental perception project and §15 documents where it fails.
* No cloud deploy, no auth, no multi-user sessions in this milestone.
* No GPU assumptions: the timing shown is whatever the host measures, and the
  UI states the device.
* No new model downloads for the UI work.

## Milestones

| # | deliverable | depends on |
|---|---|---|
| 1 | `PerceptionPipeline` lane-profile kwargs + tests | nothing |
| 2 | FastAPI app: health, config, single-image inference | 1 |
| 3 | Job runner + SSE frame stream + telemetry dump | 2 |
| 4 | Vite app: Capture and Analyse routes | 3 |
| 5 | Evaluation route rendering committed reports | 2 |
| 6 | Playwright smoke test + README section | 4, 5 |
