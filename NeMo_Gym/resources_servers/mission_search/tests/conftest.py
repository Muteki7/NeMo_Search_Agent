# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Why this file exists: the tests that need BGE-M3 (2.3 GB of memory) share ONE loaded model for the whole test run, instead of each test
loading its own copy. The model is loaded only when a test asks for the `bge_m3` fixture, and the test is skipped if the model is not on disk."""
import sys
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))
M3_PATH = SERVER_DIR.parents[2] / "Embedder" / "bge-m3"


@pytest.fixture(scope="session")
def bge_m3():
    if not (M3_PATH / "pytorch_model.bin").exists():
        pytest.skip("BGE-M3 is not in Embedder/bge-m3")
    import m3
    return m3.BGEM3(M3_PATH)
