# scripts/list_requirements.py
import subprocess
import sys
from pathlib import Path

def list_requirements(outfile: str | None = None):
    """List installed packages in the current venv and optionally write to a file."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().splitlines()

        if outfile:
            path = Path(outfile).resolve()
            path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Requirements written to {path}")
        else:
            print("\n".join(lines))
    except subprocess.CalledProcessError as e:
        print("Error running pip freeze:", e.stderr, file=sys.stderr)

if __name__ == "__main__":
    # Run: python scripts/list_requirements.py requirements.txt
    out = sys.argv[1] if len(sys.argv) > 1 else None
    list_requirements(out)
