from typing import Optional

from fastapi import HTTPException
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import User

ALGORITHM = "HS256"


async def verify_workspace_access(
    user_id: int,
    authorization: Optional[str],
    db: AsyncSession,
) -> None:
    """Raise 401/403 if workspace is password-protected and token is absent/invalid."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    if not user.password_hash:
        return  # open workspace — anyone can write
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "This workspace is password-protected")
    token = authorization[7:]
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        if int(payload.get("sub", -1)) != user_id:
            raise HTTPException(403, "Token does not belong to this workspace")
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")
