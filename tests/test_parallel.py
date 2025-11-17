import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.parallel import answer_many  # noqa: E402
from agents.tools import sample_sales  # noqa: E402


