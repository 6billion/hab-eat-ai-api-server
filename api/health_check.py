from fastapi import APIRouter
from service.health_check import health_check

router = APIRouter(
    prefix="/health-check",
    tags=["Health Check"],
)


@router.get("/")
async def health_check():
    return health_check()
