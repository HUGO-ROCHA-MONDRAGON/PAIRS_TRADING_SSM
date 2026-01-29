"""Threshold optimization module for pairs trading."""

from simul.optimization.table1 import (
    replicate_table1,
    replicate_tableS1,
    simulate_paths,
    simulate_paths_S1,
    Table1Result,
    TableS1Result,
    PAPER_TABLE1,
    NUMBA_AVAILABLE,
    _grid_search_numba,
    _grid_search_numba_S1_DE,
)

__all__ = [
    # Table 1 replication
    "replicate_table1",
    "simulate_paths",
    "Table1Result",
    "PAPER_TABLE1",
    "NUMBA_AVAILABLE",
    "_grid_search_numba",
    # Table S1 (extension)
    "replicate_tableS1",
    "TableS1Result",
    "simulate_paths_S1",
    "_grid_search_numba_S1_DE",
]
