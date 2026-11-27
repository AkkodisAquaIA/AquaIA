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
    """Raise 401/403 if the caller does not hold a valid token for this workspace.

    Unprotected workspaces (no password) allow all callers.
    Protected workspaces require a valid Bearer JWT.
    """
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "Workspace not found")
    if not user.is_protected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authentication required")
    token = authorization[7:]
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        try:
            sub = int(payload.get("sub", -1))
        except (TypeError, ValueError):
            raise HTTPException(401, "Invalid token claims")
        if sub != user_id:
            raise HTTPException(403, "Token does not belong to this workspace")
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")
