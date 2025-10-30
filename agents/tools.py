"""The executor's toolbox: a small, closed set of pandas operations. The LLM never gets to run arbitrary code;
it can only choose from these tools and supply arguments, and every argument is validated before use.
"""
import difflib

import numpy as np
import pandas as pd


