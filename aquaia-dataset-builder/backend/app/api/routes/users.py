import re
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import verify_workspace_access
from app.db.database import get_db
from app.models.models import User
from app.schemas.schemas import UserCreate, UserRead, UserUpdate, UserWithToken

router = APIRouter(prefix="/users", tags=["users"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")
    return slug or "workspace"


@router.get("", response_model=list[UserRead])
async def list_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).order_by(User.created_at))
    return result.scalars().all()


@router.post("", response_model=UserWithToken, status_code=201)
async def create_user(body: UserCreate, db: AsyncSession = Depends(get_db)):
    from app.api.routes.auth import _hash, _make_token

    username = _slugify(body.display_name)
    base = username
    counter = 2
    while await db.scalar(select(User).where(User.username == username)):
        username = f"{base}_{counter}"
        counter += 1
    user = User(username=username, display_name=body.display_name, password_hash=_hash(body.password))
    db.add(user)
    await db.flush()
    token = _make_token(user.id)
    return UserWithToken(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        is_protected=user.is_protected,
        created_at=user.created_at,
        access_token=token,
    )


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: int,
    body: UserUpdate,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(user_id, authorization, db)
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    user.display_name = body.display_name
    await db.flush()
    return user


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(user_id, authorization, db)
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    from sqlalchemy import func

    count = await db.scalar(select(func.count(User.id)))
    if count and count <= 1:
        raise HTTPException(409, "Cannot delete the last workspace")
    await db.delete(user)
    await db.flush()
