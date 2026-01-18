#!python -u
# -*- coding: utf-8 -*-

"""
Environment configuration module for turbo-genius workflows.

This module defines path variables and environment settings used throughout
the turbo-genius package.
"""

from __future__ import print_function

# python modules
import os

# set logger
from logging import getLogger

logger = getLogger("turbogenius").getChild(__name__)

# turbo-genius related path lists
turbo_genius_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
)
"""
str: Root directory of the turbo-genius package.
"""

turbo_genius_source_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
)
"""
str: Source directory of the turbo-genius package.
"""

turbo_genius_data_dir = os.path.join(turbo_genius_source_dir, "database")
"""
str: Directory containing turbo-genius database files.
"""

turbo_genius_tmp_dir = os.path.join(
    os.path.abspath(os.environ["HOME"]), ".turbo_genius_tmp"
)
"""
str: Temporary directory for turbo-genius files (default: ~/.turbo_genius_tmp).
"""

# generate pyturbo temp. dir.
os.makedirs(turbo_genius_tmp_dir, exist_ok=True)
