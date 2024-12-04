from fastapi import APIRouter
from src.service.health_check import health_check

router = APIRouter(
    prefix="/health-check",
    tags=["Health Check"],
)


@router.get("/")
def health_check():
    return health_check()
