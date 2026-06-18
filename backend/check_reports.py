import asyncio
from sqlalchemy import select, func
from app.db import AsyncSessionLocal
from app.db import AsyncSessionLocal
from app.models.user import User

async def count_users():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(func.count(User.id)))
        count = result.scalar()
        print(f"User count: {count}")
        # List email if exists
        if count > 0:
             users = await session.execute(select(User))
             print(f"User 1: {users.scalars().first().email}")

        from app.models.evidence import Report
        result_reports = await session.execute(select(func.count(Report.id)))
        report_count = result_reports.scalar()
        print(f"Report count: {report_count}")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(count_users())
