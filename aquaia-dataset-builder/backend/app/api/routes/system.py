import shutil

from fastapi import APIRouter

router = APIRouter()

@router.get("/system/storage")
async def get_storage_status():
    total, used, free = shutil.disk_usage("/app/../storage")

    used_percent = round((used / total) * 100, 2)

    return {
        "total_gb": round(total / 1024**3, 2),
        "used_gb": round(used / 1024**3, 2),
        "free_gb": round(free / 1024**3, 2),
        "used_percent": used_percent,
        "warning": used_percent >= 80,
        "critical": used_percent >= 90,
    }
