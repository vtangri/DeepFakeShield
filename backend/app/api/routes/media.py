"""
Media upload and management routes.
"""
import os
import hashlib
import aiofiles
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_async_db
from app.models import User, MediaItem
from app.core.config import settings
from app.schemas import MediaUploadResponse, MediaItemResponse
from app.api.deps import get_current_user


router = APIRouter(prefix="/media", tags=["Media"])


# Allowed file types
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/flac"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_TYPES = ALLOWED_VIDEO_TYPES | ALLOWED_AUDIO_TYPES | ALLOWED_IMAGE_TYPES


def get_media_type(mime_type: str) -> str:
    """Determine media type from MIME type."""
    if mime_type in ALLOWED_VIDEO_TYPES:
        return "video"
    elif mime_type in ALLOWED_AUDIO_TYPES:
        return "audio"
    elif mime_type in ALLOWED_IMAGE_TYPES:
        return "image"
    return "unknown"


async def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file - optimized with large chunks."""
    sha256_hash = hashlib.sha256()
    async with aiofiles.open(file_path, "rb") as f:
        # Use 1MB chunks for faster processing
        while chunk := await f.read(1024 * 1024):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


@router.post("/upload", response_model=MediaUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_media(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Upload a media file for analysis."""
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file.content_type} not allowed. Allowed types: video, audio, image"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    max_size = settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.UPLOAD_MAX_SIZE_MB}MB"
        )
    
    # Create storage directory
    storage_dir = Path(settings.STORAGE_PATH) / str(current_user.id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_ext = Path(file.filename).suffix if file.filename else ""
    unique_filename = f"{uuid4()}{file_ext}"
    file_path = storage_dir / unique_filename
    
    # Save file
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    # Compute hash
    sha256 = await compute_sha256(str(file_path))
    
    # Check for duplicate (within same user scope)
    result = await db.execute(
        select(MediaItem).where(
            MediaItem.sha256 == sha256,
            MediaItem.user_id == current_user.id
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        # Remove duplicate file
        os.remove(file_path)
        return MediaUploadResponse(
            id=existing.id,
            filename=existing.filename,
            sha256=existing.sha256,
            media_type=existing.media_type,
            duration_ms=existing.duration_ms,
            file_size=existing.file_size,
            created_at=existing.created_at
        )
    
    # Determine media type
    media_type = get_media_type(file.content_type)
    
    # Create media item
    media_item = MediaItem(
        user_id=current_user.id,
        filename=unique_filename,
        original_filename=file.filename or "unknown",
        sha256=sha256,
        file_size=file_size,
        media_type=media_type,
        mime_type=file.content_type,
        storage_path=str(file_path),
        expires_at=datetime.utcnow() + timedelta(days=30)  # 30-day retention
    )
    
    db.add(media_item)
    await db.commit()
    await db.refresh(media_item)
    
    return MediaUploadResponse(
        id=media_item.id,
        filename=media_item.filename,
        sha256=media_item.sha256,
        media_type=media_item.media_type,
        duration_ms=media_item.duration_ms,
        file_size=media_item.file_size,
        created_at=media_item.created_at
    )


@router.get("/{media_id}", response_model=MediaItemResponse)
async def get_media(
    media_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get media item details."""
    result = await db.execute(
        select(MediaItem).where(
            MediaItem.id == media_id,
            MediaItem.user_id == current_user.id
        )
    )
    media_item = result.scalar_one_or_none()
    
    if not media_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media item not found"
        )
    
    return media_item


@router.get("/", response_model=list[MediaItemResponse])
async def list_media(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List all media items for the current user."""
    result = await db.execute(
        select(MediaItem)
        .where(MediaItem.user_id == current_user.id)
        .order_by(MediaItem.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()
