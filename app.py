import os
import importlib
from fastapi import FastAPI

app = FastAPI(
    title="hab-eat-ai-api-server",
    description="hab-eat-ai-api-server",
    version="0.0.1",
    license_info={"name": "MIT License", "identifier": "MIT"},
)


for filename in os.listdir("api"):
    module = importlib.import_module("api." + filename.split(".")[0])
    print(module)
    if hasattr(module, "router"):
        app.include_router(module.router)
