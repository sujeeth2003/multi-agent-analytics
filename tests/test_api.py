import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient  # noqa: E402

from api import create_app  # noqa: E402
from agents.tools import sample_sales  # noqa: E402


