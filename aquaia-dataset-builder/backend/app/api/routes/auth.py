from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import jwt
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import ALGORITHM, verify_workspace_access
from app.core.config import settings
from app.db.database import get_db
from app.models.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _make_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(days=30)
    return jwt.encode({"sub": str(user_id), "exp": expire}, settings.secret_key, algorithm=ALGORITHM)


def _validate_pin(pin: str) -> None:
    if not pin or not pin.isdigit() or len(pin) < 6:
        raise HTTPException(422, "PIN must be at least 6 digits (numbers only)")


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


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, body.user_id)
    if not user or not user.password_hash:
        raise HTTPException(401, "Invalid credentials")
    if not _verify(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return TokenResponse(access_token=_make_token(user.id), user_id=user.id)


@router.post("/set-password", status_code=204)
async def set_password(
    body: SetPasswordRequest,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(body.user_id, authorization, db)
    user = await db.get(User, body.user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    if user.password_hash:
        if not body.old_password or not _verify(body.old_password, user.password_hash):
            raise HTTPException(401, "Current PIN is incorrect")
    _validate_pin(body.new_password)
    user.password_hash = _hash(body.new_password)
    await db.flush()


@router.delete("/password", status_code=204)
async def remove_password(
    body: RemovePasswordRequest,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(body.user_id, authorization, db)
    user = await db.get(User, body.user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    if not user.password_hash:
        return
    if not _verify(body.current_password, user.password_hash):
        raise HTTPException(401, "Current PIN is incorrect")
    user.password_hash = None
    await db.flush()


@router.get("/status/{user_id}")
async def auth_status(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    return {"user_id": user_id, "is_protected": bool(user.password_hash)}
