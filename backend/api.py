"""
api.py — FastAPI server for the SHL recommender

I kept this minimal — just a health check, a recommend endpoint,
and the root route serves the frontend HTML directly so I don't
need a separate web server for the UI.

Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommender import get_recommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising SHL Recommender...")
    try:
        get_recommender()
        logger.info("Recommender loaded ✅")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends relevant SHL Individual Test Solutions for a job query or JD.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "I am hiring for Java developers who can collaborate with business teams."
            }
        }
    }


class Assessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int] = None
    remote_support: str
    test_type: list[str]


class RecommendResponse(BaseModel):
    recommended_assessments: list[Assessment]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — returns 200 with status healthy."""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    Accept a natural language query or JD text.
    Returns 5–10 most relevant SHL Individual Test Solutions.
    """
    query = (req.query or "").strip()
    if len(query) < 5:
        raise HTTPException(status_code=400, detail="Query must be at least 5 characters.")

    logger.info(f"Query: {query[:120]}")

    try:
        rec     = get_recommender()
        results = rec.recommend(query, n_results=10)

        if not results:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        assessments = [
            Assessment(
                url              = r.get("url", ""),
                name             = r.get("name", ""),
                adaptive_support = r.get("adaptive_support", "No"),
                description      = r.get("description", ""),
                duration         = r.get("duration"),
                remote_support   = r.get("remote_support", "No"),
                test_type        = r.get("test_type", []),
            )
            for r in results
        ]

        return RecommendResponse(recommended_assessments=assessments)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/")
async def root():
    frontend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "index.html")
    if os.path.exists(frontend):
        return FileResponse(frontend, media_type="text/html")
    return JSONResponse({
        "message": "SHL Recommender API v2",
        "endpoints": {
            "health":    "GET  /health",
            "recommend": "POST /recommend  body: {query: str}",
            "docs":      "GET  /docs",
        }
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)