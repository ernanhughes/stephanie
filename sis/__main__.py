import uvicorn


def main():
    uvicorn.run(
        "sis.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["sis"],
    )
