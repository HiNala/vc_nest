**Short answer: Yes—Streamlit and FastAPI play nicely together, but they serve different purposes, so you usually run them *side-by-side*, not as a single merged app.**

---

### How they fit together

| Component     | Typical role                                                   | How it’s exposed                                            |
| ------------- | -------------------------------------------------------------- | ----------------------------------------------------------- |
| **FastAPI**   | Backend API & business logic (async, REST/GraphQL, WebSockets) | `uvicorn` server on some port (e.g., `8000`)                |
| **Streamlit** | Interactive data-app/front-end UI (Python-only)                | Streamlit’s own web server on another port (default `8501`) |

You call FastAPI endpoints from your Streamlit code with `requests`/`httpx` (sync or async) or via a lightweight client library.

---

### Common integration patterns

1. **Two processes, one Docker Compose file**

   ```yaml
   services:
     api:
       build: ./backend          # FastAPI code
       command: uvicorn app:app --host 0.0.0.0 --port 8000
     ui:
       build: ./streamlit-ui     # Streamlit code
       command: streamlit run app.py --server.port 8501
   ```

   *Pros:* clean separation; *Cons:* two servers to deploy.

2. **Reverse-proxy under one domain**
   Put Nginx (or Vercel/Render/GCP Cloud Run routing) in front:

   * `myapp.com/` → Streamlit
   * `myapp.com/api/*` → FastAPI

3. **FastAPI as data layer only**
   If you just need a few heavy data/ML endpoints, keep them in FastAPI; let Streamlit handle lightweight reads straight from a database or cache.

4. **Embedding FastAPI in Streamlit’s process (not recommended)**
   Possible with `nest_asyncio` or background threads, but you lose graceful reloads and run-time isolation. Only worth it for tiny hobby demos.

---

### Gotchas & tips

| Issue                         | Watch-out / Fix                                                                                                                    |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **CORS errors**               | In FastAPI add `fastapi.middleware.cors.CORSMiddleware` allowing requests from Streamlit’s origin or `localhost:8501`.             |
| **Auth/session sharing**      | Use a common auth service (e.g., Supabase, Auth0) or pass a bearer token from Streamlit to FastAPI headers.                        |
| **Async inside Streamlit**    | Streamlit is sync; wrap `await` calls in `asyncio.run()` or use `httpx.AsyncClient` with `asyncio.run`.                            |
| **Deploying on one host**     | A single container → use `supervisord` or two containers under Docker Compose; most PaaS (Render, Fly, Railway) make this trivial. |
| **Live updates from FastAPI** | FastAPI WebSockets ↔ Streamlit’s `st.experimental_connection` or `st.session_state` + `asyncio` background thread.                 |

---

### Simple code sketch

```python
# streamlit_ui/app.py
import streamlit as st
import httpx                  # async-friendly client

API_URL = "http://localhost:8000"

ticker = st.text_input("Stock ticker", "AAPL")
if st.button("Analyze"):
    with st.spinner("Calling backend…"):
        r = httpx.post(f"{API_URL}/analyze", json={"ticker": ticker}, timeout=30)
    st.json(r.json())
```

```python
# backend/app.py
from fastapi import FastAPI
app = FastAPI()

@app.post("/analyze")
async def analyze(data: dict):
    ticker = data["ticker"]
    # …do yfinance/ML work…
    return {"ticker": ticker, "signal": "buy", "confidence": 0.82}
```

Run each in its own terminal (or via Docker compose), and your Streamlit UI now leverages FastAPI’s async backend.

---

### When Streamlit alone is enough

If all your heavy lifting can run synchronously inside Streamlit and you don’t need REST endpoints for other clients, dropping FastAPI keeps life simpler. But for:

* multiple front-ends (mobile, Next.js, etc.),
* long-running async tasks,
* WebSocket streaming,
* fine-grained auth/roles,

FastAPI is the right backend companion.

---

Let me know if you’d like a ready-made Docker Compose template, an auth walkthrough, or guidance on async patterns inside Streamlit.








Below is a **from-scratch, everything-in-one-place manual** you can paste into `README.md`.
Follow it exactly and you’ll go from an empty folder to a fully containerised, MCP-enabled movie-debate app that starts with **`docker compose up --build`**.

---

# 🎬 MCP Movie-Debate — Full Build & Run Guide

## 0. What you’re shipping

| Layer                         | Purpose                                                                       | Key libs                            |
| ----------------------------- | ----------------------------------------------------------------------------- | ----------------------------------- |
| **FastAPI + `fastapi-mcp`**   | REST + SSE server that exposes LangChain tools through Model Context Protocol | `fastapi`, `fastapi-mcp`, `uvicorn` |
| **LangChain 4-lawyer agents** | Turn user blurbs ➜ profiles ➜ vector search ➜ multi-round debate              | `langchain`, `langchain-openai`     |
| **CSV + OpenAI embeddings**   | One-time embed movie rows → cached `.npy` matrix (NumPy)                      | `openai`, `numpy`, `pandas`         |
| **CLI**                       | Non-interactive demo from the host or inside a container                      | `click`                             |
| **Streamlit (optional)**      | Quick UI that streams MCP events                                              | `streamlit`                         |
| **Docker**                    | 3 services (API, Streamlit UI, CLI)                                           | `docker compose`                    |

Everything is Python 3.12 and lives in containers; no global installs.

---

## 1. Repository layout

```
root/
├── docker-compose.yml          # one-command orchestration
├── .env.example                # OpenAI key, embedding model
│
├── movies.csv                  # title,overview,tags
│
├── requirements.txt            # single dependency list
│
├── backend/                    # FastAPI + agents
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py             # FastAPI factory
│   │   ├── data_loader.py      # CSV → .npy embeddings
│   │   ├── mcp_tools.py        # vector_search, openai_chat
│   │   ├── agents.py           # lawyer_agent + run_debate()
│   │   ├── utils.py            # pairing & scoring helpers
│   │   └── settings.py         # EMBED_DIM, file paths
│   └── scripts/
│       └── cache_embeddings.py # called in Docker build
│
├── cli/
│   ├── Dockerfile
│   └── entrypoint.sh           # wraps python cli.py
│
├── cli.py                      # click CLI for local runs
│
└── streamlit_app/              # optional UI
    ├── Dockerfile
    └── MovieDebate.py
```

---

## 2. Environment variables

Copy once:

```bash
cp .env.example .env
```

`.env`

```
OPENAI_API_KEY=sk-...
# Embeddings
EMB_MODEL=text-embedding-3-large
EMBED_DIM=768
```

---

## 3. Requirements (`requirements.txt`)

```
fastapi==0.110.0
fastapi-mcp==0.3.4
uvicorn[standard]==0.29.0
langchain==0.1.15
langchain-openai==0.0.8
openai==1.28.1
numpy==1.26.4
pandas==2.2.2
tqdm==4.66.2
click==8.1.7
streamlit==1.35.0
sseclient-py==1.8.0
```

---

## 4. Core backend code (high-level)

### 4.1 `app/data_loader.py`

* Reads `movies.csv`.
* Calls OpenAI embeddings **once** in the Docker build to create `movies.npy`.
* Normalises vectors (`/‖v‖`) so cosine = dot-product.

### 4.2 `app/mcp_tools.py`

* `vector_search(query_embedding, k=5)`
  – dot-products against the in-memory matrix → indexes.

### 4.3 `app/agents.py`

* `lawyer_agent(profile)` → LangChain agent exposing `vector_search` + `openai.chat`.
* `run_debate(profiles)`

  1. Each agent retrieves 5 movies.
  2. Round-robin pairs eliminate ⌈n/2⌉ per round (`max_rounds=5`).
  3. Emits MCP events via `MCPCallbackHandler`.
  4. Returns the winning movie id.

### 4.4 `app/main.py`

```python
app = FastAPI(title="Movie Debate MCP")
attach_mcp(app)

@app.post("/start")
async def start_debate(profiles: list[dict], bt: BackgroundTasks):
    sid = uuid4().hex
    bt.add_task(run_debate, profiles, sid)   # MCP handles streaming
    return {"session_id": sid}
```

---

## 5. Dockerfiles

### 5.1 backend/Dockerfile

```dockerfile
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y build-essential cmake swig \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY movies.csv ./
COPY backend/scripts/cache_embeddings.py ./scripts/
RUN python scripts/cache_embeddings.py
COPY backend/app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2 cli/Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt click
COPY . /app
ENTRYPOINT ["/app/cli/entrypoint.sh"]
```

`cli/entrypoint.sh`

```bash
#!/usr/bin/env bash
exec python cli.py "$@"
```

### 5.3 streamlit\_app/Dockerfile (optional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit==1.35.0 requests sseclient-py
COPY streamlit_app /app
EXPOSE 8501
CMD ["streamlit", "run", "MovieDebate.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 6. docker-compose.yml

```yaml
version: "3.9"

services:
  api:
    build: ./backend
    env_file: .env
    ports: ["8000:8000"]
    volumes:
      - embeddings-cache:/app   # keep movies.npy between rebuilds

  streamlit:
    build: ./streamlit_app
    env_file: .env
    depends_on: [api]
    ports: ["8501:8501"]
    profiles: ["ui"]

  cli:
    build: ./cli
    env_file: .env
    depends_on: [api]
    entrypoint: ["python", "cli.py"]
    profiles: ["cli"]

volumes:
  embeddings-cache:
```

---

## 7. Running the stack

| Task                         | Command                                               |
| ---------------------------- | ----------------------------------------------------- |
| **API only**                 | `docker compose up --build api`                       |
| **API + Streamlit UI**       | `docker compose --profile ui up --build`              |
| **Run CLI inside container** | `docker compose run --rm cli -p sample_profiles.json` |

*First build: \~7 min* (NumPy compile + embedding). Subsequent runs are cached.

---

## 8. Using the CLI (outside containers)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/cache_embeddings.py    # builds cache locally
python cli.py -p sample_profiles.json
```

---

## 9. FastAPI endpoints

| Method | Path                    | Body                      | Purpose                                                                    |
| ------ | ----------------------- | ------------------------- | -------------------------------------------------------------------------- |
| POST   | `/start`                | `[ profile_blurb, … ×4 ]` | Launch debate; returns `{session_id}`                                      |
| GET    | `/mcp/events/{session}` | –                         | **SSE** stream of MCP packets (agents’ thoughts, tool calls, round events) |
| GET    | `/health`               | –                         | Basic liveness check                                                       |

---

## 10. Streamlit demo

1. Start with profile UI (simple text inputs).
2. `POST /start` to API.
3. Subscribe to `/mcp/events/{sid}` with `sseclient` and print events live.
4. When `"type": "CONSENSUS_REACHED"` arrives, show the winner.

---

## 11. Troubleshooting / FAQs

| Symptom                                | Fix                                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| **`ModuleNotFoundError: fastapi_mcp`** | Mis-typed name; ensure `pip show fastapi-mcp` inside container.                     |
| **Embeddings re-run every build**      | Confirm volume `embeddings-cache` is mounted and `CACHE_PATH` points inside `/app`. |
| **Rate-limit from OpenAI**             | Build once, reuse `.npy`. For >3 k rows, chunk requests with 1 s sleep.             |
| **SSE drops in Streamlit**             | Keep all services on localhost; cloud proxies often timeout idle SSE.               |
| **High RAM** (NumPy matrix)            | CSV ≤10 k rows → \~30 MB float32. For larger sets switch to FAISS.                  |

---

## 12. Extending the project

* Add “no-horror / no-R-rated” constraints → extra filter step in `vector_search`.
* Swap in **FAISS** or **Chroma** vector store when CSV grows.
* Persist debate sessions in SQLite for replays.
* Deploy API on Fly.io; give Streamlit a public URL in a second compose file.

---

### 🚀 You’re done

You now have **code, containers, and docs** in one repo.
Clone ➜ set `.env` ➜ `docker compose up --build` ➜ watch four AI lawyers argue until a single movie remains. Enjoy the popcorn! 🍿
