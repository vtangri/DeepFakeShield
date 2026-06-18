import asyncio
import uuid
from datetime import datetime, timedelta
import random
from app.db import AsyncSessionLocal
from app.models import User, MediaItem, AnalysisJob, Report

async def seed_data():
    async with AsyncSessionLocal() as session:
        # 1. Get or Create User
        result = await session.execute(select(User))
        user = result.scalars().first()
        if not user:
            print("No user found. Please register a user first.")
            return
            
        print(f"Seeding data for user: {user.email}")
        
        # 2. Create Media Items & Jobs
        media_types = ['video', 'audio', 'image']
        
        for i in range(20):
            m_type = random.choice(media_types)
            is_fake = random.random() > 0.6
            score = random.uniform(0.7, 0.99) if is_fake else random.uniform(0.0, 0.3)
            if random.random() > 0.8: # Some suspicious
                score = random.uniform(0.35, 0.65)
                
            media = MediaItem(
                id=uuid.uuid4(),
                user_id=user.id,
                filename=f"sample_{m_type}_{i}.{m_type == 'image' and 'jpg' or 'mp4'}",
                original_filename=f"test_file_{i}.{m_type}",
                media_type=m_type,
                mime_type=f"{m_type}/{'jpeg' if m_type == 'image' else 'mp4'}",
                file_size=1024 * 1024 * random.randint(1, 50),
                sha256=f"hash_{i}",
                storage_path=f"/tmp/sample_{i}"
            )
            session.add(media)
            
            job = AnalysisJob(
                id=uuid.uuid4(),
                media_id=media.id,
                status="DONE",
                stage="done",
                progress=100.0,
                overall_score=score,
                label="FAKE" if score > 0.7 else ("SUSPICIOUS" if score > 0.3 else "AUTHENTIC"),
                started_at=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                completed_at=datetime.utcnow() - timedelta(days=random.randint(0, 30))
            )
            session.add(job)
            
            report = Report(
                id=uuid.uuid4(),
                job_id=job.id,
                summary="Auto-generated seed report",
                full_report={"mock": True},
                generated_at=job.completed_at
            )
            session.add(report)
            
        await session.commit()
        print("Seeding complete!")

if __name__ == "__main__":
    import sys
    import os
    from sqlalchemy import select
    sys.path.append(os.getcwd())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(seed_data())
