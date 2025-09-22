from enum import Enum
from typing import List

from shapely import LineString, Point


class CurveType(Enum):
    """
    Enum for different turning maneuvers.
    """

    U_TURN = 0
    PI_CURVE = 1
    HOOK = 2
    LOOP = 3
    UNDEFINED = 4


class Curve:
    """
    Wrapper class for LineString with additional data on how it was constructed.
    """

    def __init__(
            self,
            path: List[Point] | LineString,
            curve_type: CurveType,
            valid: bool):
        self.curve_type = curve_type
        self.valid = valid
        self.path: LineString = None
        if isinstance(path, list):
            self.path = LineString(path)
        elif isinstance(path, LineString):
            self.path = path
        else:
            raise TypeError("Curve must be constructed with List[Point] or LineString")

    @staticmethod
    def get_dominant_curve_type(curve1, curve2) -> CurveType:
        """
        Returns the 'dominant' type of two given curves.

        Hooks, Loops and PI-Turns are considered 'dominant' over U-Turn or Undefined.
        """
        if CurveType.HOOK in [curve1.curve_type, curve2.curve_type]:
            return CurveType.HOOK
        elif CurveType.LOOP in [curve1.curve_type, curve2.curve_type]:
            return CurveType.LOOP
        elif CurveType.PI_CURVE in [curve1.curve_type, curve2.curve_type]:
            return CurveType.PI_CURVE
        elif CurveType.U_TURN in [curve1.curve_type, curve2.curve_type]:
            return CurveType.U_TURN
        else:
            return CurveType.UNDEFINED
