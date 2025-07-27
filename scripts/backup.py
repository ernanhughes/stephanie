import os
import subprocess
from datetime import datetime

# SELECT pg_size_pretty(pg_database_size('co'));

def backup_postgres_db(
    db_name="co",
    user="co",
    host="localhost",
    port="5432",
    backup_dir="F:/backups",
    file_prefix="stephanie_backup",
    password_env_var="PGPASSWORD"
):
    """
    Creates a timestamped backup of a PostgreSQL database using pg_dump.

    Args:
        db_name (str): Name of the PostgreSQL database to back up.
        user (str): PostgreSQL username.
        host (str): Database host.
        port (str): Database port.
        backup_dir (str): Directory to store the backup files.
        file_prefix (str): Prefix for the backup file name.
        password_env_var (str): Name of the environment variable storing the DB password.
    """
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{file_prefix}_{timestamp}.sql"
    filepath = os.path.join(backup_dir, filename)

    # password = os.getenv(password_env_var)
    # if not password:
    #     print(f"[ERROR] Environment variable '{password_env_var}' not set.")
    #     return

    env = os.environ.copy()
    env["PGPASSWORD"] = "co"

    cmd = [
        "E:\\Program Files\\PostgreSQL\\16\\bin\\pg_dump.exe",
        "-U", user,
        "-h", host,
        "-p", port,
        "-F", "c",  # custom format
        "-f", filepath,
        db_name
    ]

    try:
        print (cmd)
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip())

        process.wait()
        if process.returncode == 0:
            print(f"[✔] Backup successful: {filepath}")
        else:
            print("[✖] Backup failed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Backup failed: {e}")

if __name__ == "__main__":
    backup_postgres_db()
