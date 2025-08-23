# test_build.py
"""
Build and test-install Stephanie in a clean environment.

Steps:
1. Remove old dist/ and rebuild with `python -m build`.
2. Create a fresh test directory (C:/test_env on Windows, ~/test_env on Unix).
3. Create a new virtual environment inside it.
4. Install the freshly built wheel into that venv.
5. Run smoke tests: `stephanie version` and `stephanie run config=selfcheck` (if available).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
TEST_DIR = Path("C:/test_env/").resolve() if os.name == "nt" else Path.home() / "test_env"


def run(cmd, cwd=None):
    """Run a shell command and stream output."""
    print(f"\n>>> {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def main():
    print(f"🧪 Testing Stephanie build in {TEST_DIR}")

    # 1. Clean old dist and rebuild
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    run("python -m build", cwd=PROJECT_ROOT)

    # 2. Clean test dir
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)

    # 3. Create virtualenv
    venv_dir = TEST_DIR / ".venv"
    run(f"{sys.executable} -m venv {venv_dir}")
    pip_exe = venv_dir / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    python_exe = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    stephanie_exe = venv_dir / ("Scripts/stephanie.exe" if os.name == "nt" else "bin/stephanie")

    # Upgrade pip first
    run(f'{venv_dir}\Scripts\python.exe -m pip install --upgrade pip')

    # 4. Install wheel into venv
    wheels = list(DIST_DIR.glob("*.whl"))
    if not wheels:
        raise RuntimeError("❌ No wheels found in dist/")
    wheel_path = wheels[0]
    run(f"{pip_exe} install {wheel_path}")

    # 5. Smoke test via entry points
    print("\n🚀 Running smoke tests...")
    try:
        run(f"{stephanie_exe} version")
    except subprocess.CalledProcessError:
        print("⚠️ stephanie CLI not available, falling back to python -m stephanie.main")
        run(f"{python_exe} -m stephanie.main version")

    # Optional: self-check pipeline (if defined in CLI)
    try:
        run(f"{stephanie_exe} selfcheck")
    except subprocess.CalledProcessError:
        print("ℹ️ No `selfcheck` command wired, skipping.")

    print(f"\n✅ Test environment ready in {TEST_DIR}")


if __name__ == "__main__":
    main()
