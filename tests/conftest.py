"""Pytest configuration for nanochat tests."""
import sys
from pathlib import Path

# Add project root to Python path so we can import from scripts/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
