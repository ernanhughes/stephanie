import uvicorn
import os


def main():
    uvicorn.run(
        "sis.main:app",
        # Keep local development private by default. Remote exposure requires
        # an explicitly configured host and an independently verified auth layer.
        host=os.environ.get("SIS_HOST", "127.0.0.1"),
        port=int(os.environ.get("SIS_PORT", "8000")),
        reload=True,
        reload_dirs=["sis"],
    )


if __name__ == "__main__":
    main()
