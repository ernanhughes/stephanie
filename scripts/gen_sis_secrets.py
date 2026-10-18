"""Generate SIS authentication secrets for Docker Compose.

Docker Compose interpolates ``$`` in ``env_file`` values, and a PBKDF2 password
hash is built from ``$``-separated fields. This helper prints the values with
every ``$`` doubled as ``$$`` so Compose delivers the hash unchanged.

Usage::

    # Prompt for the password and write the file:
    python scripts/gen_sis_secrets.py --append .env.sis

    # Non-interactive (CI/dev fixture):
    python scripts/gen_sis_secrets.py --password 'my-password' --append .env.sis

Then start the stack (see docs/docker.md)::

    docker compose up -d --build
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import sys

from sis.security import hash_password


def build_secret_block(password: str) -> str:
    escaped = hash_password(password).replace("$", "$$")
    session = secrets.token_urlsafe(48)
    return f"SIS_AUTH_PASSWORD_HASH={escaped}\nSIS_SESSION_SECRET={session}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate .env.sis for Docker Compose")
    parser.add_argument("--password", help="password (omitted: prompt without echo)")
    parser.add_argument("--append", help="write to this file instead of stdout")
    args = parser.parse_args()

    if args.password:
        password = args.password
    else:
        password = getpass.getpass("SIS local password: ")
        if password != getpass.getpass("Confirm password: "):
            print("Passwords do not match", file=sys.stderr)
            return 2
    if not password:
        print("Password must not be empty", file=sys.stderr)
        return 2

    block = build_secret_block(password)
    if args.append:
        with open(args.append, "w", encoding="utf-8") as handle:
            handle.write(block)
        print(f"wrote {args.append} (every '$' escaped as '$$')")
    else:
        sys.stdout.write(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
