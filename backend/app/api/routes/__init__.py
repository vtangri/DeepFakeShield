"""Routes module exports."""
from .auth import router as auth_router
from .media import router as media_router
from .analysis import router as analysis_router
from .evidence import router as evidence_router
from .reports import router as reports_router

__all__ = [
    "auth_router",
    "media_router",
    "analysis_router",
    "evidence_router",
    "reports_router",
]
