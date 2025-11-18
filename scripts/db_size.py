import psycopg2
import os
from psycopg2 import sql
import shutil

def get_terminal_width():
    """Get terminal width for dynamic bar sizing"""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80

def format_bytes(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_database_size(cursor):
    """Get total database size"""
    cursor.execute("SELECT pg_database_size(current_database());")
    return cursor.fetchone()[0]

def get_table_sizes(cursor):
    """Get all table sizes with schema names"""
    cursor.execute("""
        SELECT 
            schemaname || '.' || tablename AS full_table_name,
            pg_total_relation_size(schemaname || '.' || tablename) AS size_bytes
        FROM pg_tables
        WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
        ORDER BY size_bytes DESC;
    """)
    return cursor.fetchall()

def draw_bar_graph(current_size, max_size, max_width=50):
    """Draw a proportional bar graph"""
    if max_size == 0:
        return ""
    ratio = current_size / max_size
    bar_length = int(ratio * max_width)
    return "█" * bar_length

def main():
    # Database connection parameters (use environment variables for security)
    conn_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'co'),
        'user': os.getenv('DB_USER', 'co'),
        'password': 'co'
    }

    try:
        # Connect to database
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Get total database size
        total_size = get_database_size(cursor)
        print(f"📊 DATABASE SIZE: {format_bytes(total_size)}\n")
        
        # Get table sizes
        tables = get_table_sizes(cursor)
        if not tables:
            print("No user tables found.")
            return
            
        # Find largest table for scaling
        max_table_size = tables[0][1] if tables else 0
        terminal_width = get_terminal_width()
        max_bar_width = min(50, terminal_width - 40)  # Leave space for labels
        
        print("TABLE SIZES (Largest first):")
        print("-" * terminal_width)
        
        for table_name, size_bytes in tables:
            formatted_size = format_bytes(size_bytes)
            percentage = (size_bytes / total_size) * 100 if total_size > 0 else 0
            bar = draw_bar_graph(size_bytes, max_table_size, max_bar_width)
            
            # Truncate long table names to fit terminal
            max_name_length = terminal_width - len(formatted_size) - len(f"({percentage:.1f}%)") - max_bar_width - 5
            display_name = (table_name[:max_name_length-3] + '...') if len(table_name) > max_name_length else table_name
            
            print(f"{display_name:<{max_name_length}} {bar} {formatted_size} ({percentage:.1f}%)")
        
        # Close connections
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()