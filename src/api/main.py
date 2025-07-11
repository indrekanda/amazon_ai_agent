from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import AsyncClient
import logging
from contextlib import asynccontextmanager
from api.core.config import settings
from api.api.middleware import RequestIdMiddleware
from api.api.endpoints import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

client = AsyncClient(timeout=settings.DEFAULT_TIMEOUT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application is starting up...")
    yield
    logger.info("Application is shutting down...")
    await client.aclose()

app = FastAPI(lifespan=lifespan)

# Request ID Middleware
app.add_middleware(RequestIdMiddleware)

# To communicate with streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "API"}

# @app.get("/health")
# async def health():
#     return {"status": "ok"}