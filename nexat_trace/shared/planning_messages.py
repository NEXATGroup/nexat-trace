from enum import Enum


class PlanningMsg(Enum):
    """
    Enum for different planner warnings & messages.
    """

    # If there was an error during the calculation of curves
    CURVE_CALCULATION_ERROR = 0

    # If your ab lines have more than 2 vertices
    EXTRA_AB_LINE_POINTS_FOUND = 1

    # If the target turning headland was not available on the track system
    TARGET_HEADLAND_NOT_AVAILABLE = 2

    # If there is a severe route error with jumps violating the track graph rules
    NON_DRIVABLE_ROUTE = 3

    # If there are headland track rings that are too small to be rounded to the configured turning radius
    CUTOUT_RING_SMOOTHING_ERROR = 4

    # The route optimization was changed to round trip optimization because configured start and finish locations were the same
    ROUTE_CHANGED_TO_ROUND_TRIP = 5

    # Route finish point was changed because it was on the same ab line as the start
    CHANGED_END_POINT = 6

    # If there is a part of the path that is less than working width / 2 from the field border
    COLLISION = 7

    # Path segmentation when there shouldn't be one, Path will be trimmed
    UNEXPECTED_SEGMENTATION = 8

    # Path segmentation without direction change or start and end of segments are not the same
    NO_DIRECTION_CHANGE_AT_SEGMENTATION = 9

