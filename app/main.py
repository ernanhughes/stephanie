from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import api

from fastapi.staticfiles import StaticFiles

app = FastAPI()
# Mount API
app.include_router(api.router, prefix="/api")

# Allow your frontend (e.g., jQuery served from file:// or localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
def root():
    return {"status": "Stephanie is alive"}
