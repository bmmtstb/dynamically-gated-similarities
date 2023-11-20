"""
Predefined constants that will not change and might be used at different places.
"""
import os

PRINT_PRIORITY: list[str] = ["none", "normal", "debug", "all"]

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
