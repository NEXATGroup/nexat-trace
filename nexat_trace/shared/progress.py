from enum import Enum

"""
This module is meant to be a global singleton for managing
the route planning progress
"""


class PlanningStage(Enum):
    """
    Enum for stages during planning.
    """

    IDLE = "Idle"
    TRACK_GRAPH = "Preparing field"
    NET_GRAPH = "Preparing optimization"
    PLANNING_ROUTE = "Searching for the best route"
    PLANNING_CURVES = "Calculating curves"


class PlanningProgress:
    """
    Class holding info about the current planning progress.
    """

    planning_percent = 0
    planning_stage = PlanningStage.IDLE
