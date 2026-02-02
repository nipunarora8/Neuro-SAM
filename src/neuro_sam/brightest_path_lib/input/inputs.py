"""Enums for the types of inputs for the cost, heuristic and search functions.
"""

from enum import Enum

class CostFunction(Enum):
	RECIPROCAL = "reciprocal"

class HeuristicFunction(Enum):
	EUCLIDEAN = "euclidean"

class SearchFunction(Enum):
    ASTAR = "astar"
    NBASTAR = "nbastar"
