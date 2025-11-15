"""Answer cache: the same question on the same data should not run the whole agent graph twice.

Redis is the shared cache (all API workers see the same entries). With no Redis configured, a dict is used so the
project still runs on a laptop.  The key includes a fingerprint of the data, so new data never returns old answers.
"""
import hashlib
import json
import os


