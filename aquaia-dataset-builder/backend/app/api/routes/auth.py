from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import ALGORITHM
from app.core.config import settings
from app.db.database import get_db
from app.models.models import User

router = APIRouter(prefix="/auth", tags=["auth"])
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


class LoginRequest(BaseModel):
    user_id: int
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int


class SetPasswordRequest(BaseModel):
    user_id: int
    new_password: str
    old_password: Optional[str] = None


class RemovePasswordRequest(BaseModel):
    user_id: int
    current_password: str


def _make_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(days=30)
    return jwt.encode({"sub": str(user_id), "exp": expire}, settings.secret_key, algorithm=ALGORITHM)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, body.user_id)
    if not user or not user.password_hash:
        raise HTTPException(401, "Invalid credentials")
    if not _pwd.verify(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return TokenResponse(access_token=_make_token(user.id), user_id=user.id)


@router.post("/set-password", status_code=204)
async def set_password(body: SetPasswordRequest, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, body.user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    if user.password_hash:
        if not body.old_password or not _pwd.verify(body.old_password, user.password_hash):
            raise HTTPException(401, "Current password is incorrect")
    if not body.new_password:
        raise HTTPException(400, "New password cannot be empty")
    user.password_hash = _pwd.hash(body.new_password)
    await db.flush()


@router.delete("/password", status_code=204)
async def remove_password(body: RemovePasswordRequest, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, body.user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    if not user.password_hash:
        return
    if not _pwd.verify(body.current_password, user.password_hash):
        raise HTTPException(401, "Current password is incorrect")
    user.password_hash = None
    await db.flush()


@router.get("/status/{user_id}")
async def auth_status(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    return {"user_id": user_id, "is_protected": bool(user.password_hash)}
