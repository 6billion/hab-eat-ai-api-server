import os
import importlib
from fastapi import FastAPI

app = FastAPI(
    title="hab-eat-ai-api-server",
    description="hab-eat-ai-api-server",
    version="0.0.1",
    license_info={"name": "MIT License", "identifier": "MIT"},
)


for filename in os.listdir("src/api"):
    if ".py" not in filename:
        continue
    module = importlib.import_module("src.api." + filename.split(".")[0])
    if hasattr(module, "router"):
        app.include_router(module.router)
