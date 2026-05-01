"""pytest configuration for 09-skills.

Adds the package root to sys.path so that skill_base, skilled_agent,
skill_test_runner, and the skills.* subpackage are all importable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
