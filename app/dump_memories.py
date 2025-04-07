from app.db_utils import connect_db
import logging

logger = logging.getLogger(__name__)

def main():
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("SELECT title, user_text, ai_text FROM memory ORDER BY timestamp DESC")
    rows = cur.fetchall()

    for title, user, ai in rows:
        print("====", title, "====")
        print("USER:")
        print(user.strip())
        print("\nASSISTANT:")
        print(ai.strip())
        print("\n" + "-" * 50 + "\n")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
