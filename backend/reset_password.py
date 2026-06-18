import asyncio
from sqlalchemy import select
from app.db import AsyncSessionLocal
from app.models import User
from app.core import get_password_hash

async def reset_password():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.email == "test@example.com"))
        user = result.scalar_one_or_none()
        
        if not user:
            print("User test@example.com not found!")
            return

        print(f"Resetting password for {user.email}...")
        user.hashed_password = get_password_hash("password")
        session.add(user)
        await session.commit()
        print("Password reset to 'password' successfully.")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(reset_password())
