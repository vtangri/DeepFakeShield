"""
Celery application configuration.
"""
from celery import Celery
from .config import settings


celery_app = Celery(
    "deepfakeshield",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.preprocess",
        "app.workers.inference",
        "app.workers.report",
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "app.workers.preprocess.*": {"queue": "preprocess"},
        "app.workers.inference.*": {"queue": "inference"},
        "app.workers.report.*": {"queue": "default"},
    },
    
    # Retry policy
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Rate limiting
    worker_prefetch_multiplier=1,
    
    # Result expiration (24 hours)
    result_expires=86400,
)

# Task state definitions
class TaskState:
    """Analysis job states."""
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    EXTRACTING = "EXTRACTING"
    TRANSCRIBING = "TRANSCRIBING"
    INFER_VIDEO = "INFER_VIDEO"
    INFER_AUDIO = "INFER_AUDIO"
    LIPSYNC = "LIPSYNC"
    FUSION = "FUSION"
    REPORT = "REPORT"
    DONE = "DONE"
    FAILED = "FAILED"


# State machine transitions
STATE_TRANSITIONS = {
    TaskState.PENDING: [TaskState.VALIDATING, TaskState.FAILED],
    TaskState.VALIDATING: [TaskState.EXTRACTING, TaskState.FAILED],
    TaskState.EXTRACTING: [TaskState.TRANSCRIBING, TaskState.FAILED],
    TaskState.TRANSCRIBING: [TaskState.INFER_VIDEO, TaskState.FAILED],
    TaskState.INFER_VIDEO: [TaskState.INFER_AUDIO, TaskState.FAILED],
    TaskState.INFER_AUDIO: [TaskState.LIPSYNC, TaskState.FAILED],
    TaskState.LIPSYNC: [TaskState.FUSION, TaskState.FAILED],
    TaskState.FUSION: [TaskState.REPORT, TaskState.FAILED],
    TaskState.REPORT: [TaskState.DONE, TaskState.FAILED],
    TaskState.DONE: [],
    TaskState.FAILED: [],
}
