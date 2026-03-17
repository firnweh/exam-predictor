"""Load data into SQLite and launch the dashboard."""
import subprocess
import sys


def main():
    from utils.db import init_db
    from utils.loader import load_all_extracted

    db_path = "data/exam.db"

    print("Initializing database...")
    init_db(db_path)

    print("Loading extracted questions...")
    total = load_all_extracted(db_path)

    if total == 0:
        print("\nNo questions found in data/extracted/")
        print("Add JSON files extracted via the prompt template, then run again.")
        return

    print(f"\nLoaded {total} questions into {db_path}")
    print("\nLaunching dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()
