"""Spatial indexing: candidate-pair generation and point (bbox) queries."""

from .pairs import PairFinder, available_backends, make_pair_finder
from .points import PointIndex, build_point_index

__all__ = ["PairFinder", "PointIndex", "available_backends", "build_point_index", "make_pair_finder"]
