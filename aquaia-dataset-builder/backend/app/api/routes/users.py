import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.models import User
from app.schemas.schemas import UserCreate, UserRead, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")
    return slug or "workspace"


@router.get("", response_model=list[UserRead])
async def list_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).order_by(User.created_at))
    return result.scalars().all()


@router.post("", response_model=UserRead, status_code=201)
async def create_user(body: UserCreate, db: AsyncSession = Depends(get_db)):
    username = _slugify(body.display_name)
    # Ensure unique username
    base = username
    counter = 2
    while await db.scalar(select(User).where(User.username == username)):
        username = f"{base}_{counter}"
        counter += 1
    user = User(username=username, display_name=body.display_name)
    db.add(user)
    await db.flush()
    return user


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(user_id: int, body: UserUpdate, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    user.display_name = body.display_name
    await db.flush()
    return user


@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    from sqlalchemy import func

    count = await db.scalar(select(func.count(User.id)))
    if count and count <= 1:
        raise HTTPException(409, "Cannot delete the last workspace")
    await db.delete(user)
    await db.flush()
