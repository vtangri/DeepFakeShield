"""
Pytest configuration and fixtures.
"""
import pytest
import os
import tempfile
from pathlib import Path
# Set test environment (must be done before importing app modules)
os.environ["TESTING"] = "1"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["SECRET_KEY"] = "test-secret-key"
# Use a temporary directory for storage during tests to avoid permission issues and cleanup
os.environ["STORAGE_PATH"] = str(Path(tempfile.gettempdir()) / "deepfakeshield_test_storage")

from fastapi.testclient import TestClient
from app.db.session import async_engine, AsyncSessionLocal
from app.db.base import Base
# Import all models to ensure they are registered with Base.metadata
from app.models import User, MediaItem, AnalysisJob, Segment, ModelRun, EvidenceArtifact, Report, AuditLog


@pytest.fixture(scope="session")
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def sample_video(temp_storage):
    """Create a sample video file for testing."""
    video_path = temp_storage / "sample.mp4"
    # Write minimal MP4 header (not a valid video but enough for file type checks)
    with open(video_path, "wb") as f:
        f.write(b'\x00\x00\x00\x1c\x66\x74\x79\x70\x6d\x70\x34\x32')
        f.write(b'\x00' * 1000)
    return video_path


@pytest.fixture(scope="function")
def sample_audio(temp_storage):
    """Create a sample audio file for testing."""
    audio_path = temp_storage / "sample.wav"
    # Write minimal WAV header
    import struct
    with open(audio_path, "wb") as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 44))  # File size
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Chunk size
        f.write(struct.pack('<H', 1))   # Audio format (PCM)
        f.write(struct.pack('<H', 1))   # Channels
        f.write(struct.pack('<I', 16000))  # Sample rate
        f.write(struct.pack('<I', 32000))  # Byte rate
        f.write(struct.pack('<H', 2))   # Block align
        f.write(struct.pack('<H', 16))  # Bits per sample
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', 0))
    return audio_path


@pytest.fixture(scope="function") 
def sample_frames(temp_storage):
    """Create sample frame images for testing."""
    frames_dir = temp_storage / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    try:
        import numpy as np
        import cv2
        
        for i in range(5):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), frame)
    except ImportError:
        # Create empty placeholder files if cv2 not available
        for i in range(5):
            (frames_dir / f"frame_{i:04d}.jpg").touch()
    
    return frames_dir


@pytest.fixture(scope="session", autouse=True)
async def init_db():
    """Initialize test database."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session():
    """Get async db session."""
    async with AsyncSessionLocal() as session:
        yield session


@pytest.fixture
def client(db_session):
    """Create test client with database dependency override."""
    from app.main import app
    from app.api.deps import get_async_db
    
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_async_db] = override_get_db
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()
