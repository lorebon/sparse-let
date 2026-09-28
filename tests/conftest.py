"""Keep tiny regression tests fast without changing a user's environment."""

import pytest
from numba import get_num_threads, set_num_threads


@pytest.fixture(scope="session", autouse=True)
def numba_threads():
    previous = get_num_threads()
    set_num_threads(min(2, previous))
    yield
    set_num_threads(previous)
