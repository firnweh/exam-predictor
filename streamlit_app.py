"""Entry point for Streamlit Cloud deployment — redirects to dashboard/app.py"""
import sys
import os
import importlib.util

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Load and run the actual dashboard
spec = importlib.util.spec_from_file_location("app", os.path.join(ROOT, "dashboard", "app.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
