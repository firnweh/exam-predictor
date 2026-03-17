"""Entry point for Streamlit Cloud — executes dashboard/app.py directly."""
import sys, os

# Ensure project root is on sys.path so `from analysis.xxx import ...` works
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Execute the real dashboard in this process (Streamlit Cloud compatible)
exec(open(os.path.join(ROOT, "dashboard", "app.py")).read())
