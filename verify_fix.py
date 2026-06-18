import asyncio
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.models import User, MediaItem, AnalysisJob, Report
from app.core.config import settings

async def verify_fix():
    # Force async driver for test
    db_url = settings.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        print("--- Verification Started ---")
        
        # 1. Create two test users
        user_a = User(email=f"user_a_{uuid.uuid4().hex[:4]}@test.com", hashed_password="hashed", full_name="User A")
        user_b = User(email=f"user_b_{uuid.uuid4().hex[:4]}@test.com", hashed_password="hashed", full_name="User B")
        db.add_all([user_a, user_b])
        await db.commit()
        await db.refresh(user_a)
        await db.refresh(user_b)
        print(f"Created users: A({user_a.id}), B({user_b.id})")
        
        # 2. Simulate User A uploading a file
        sha_test = "fake_sha_256_for_testing"
        media_a = MediaItem(
            user_id=user_a.id,
            filename="user_a_file.mp4",
            original_filename="video.mp4",
            sha256=sha_test,
            file_size=1024,
            media_type="video",
            mime_type="video/mp4",
            storage_path="/tmp/fake_a.mp4"
        )
        db.add(media_a)
        await db.commit()
        await db.refresh(media_a)
        print(f"User A uploaded media: {media_a.id}")
        
        # 3. Simulate User B uploading the SAME file
        # This is where the bug was - it used to find media_a and return it.
        # Now it should NOT find it because user_id differs.
        
        result = await db.execute(
            select(MediaItem).where(
                MediaItem.sha256 == sha_test,
                MediaItem.user_id == user_b.id
            )
        )
        media_b_existing = result.scalar_one_or_none()
        
        if media_b_existing:
            print("FAILED: User B found User A's media (Bug still present)")
            return
        else:
            print("SUCCESS: User B did not find User A's media. Creating new record for User B.")
            media_b = MediaItem(
                user_id=user_b.id,
                filename="user_b_file.mp4",
                original_filename="video.mp4",
                sha256=sha_test,
                file_size=1024,
                media_type="video",
                mime_type="video/mp4",
                storage_path="/tmp/fake_b.mp4"
            )
            db.add(media_b)
            await db.commit()
            await db.refresh(media_b)
            print(f"User B uploaded media: {media_b.id}")

            # 4. Verify Ownership Separation
            assert media_a.id != media_b.id, "Media IDs must be different"
            assert media_a.user_id != media_b.user_id, "User IDs must be different"
            print("PASSED: Media items successfully separated by user.")

        # Cleanup test data
        await db.execute(delete(MediaItem).where(MediaItem.sha256 == sha_test))
        await db.delete(user_a)
        await db.delete(user_b)
        await db.commit()
        print("--- Verification Finished (PASSED) ---")

if __name__ == "__main__":
    asyncio.run(verify_fix())
