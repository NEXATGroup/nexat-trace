from math import pi
from typing import Dict, List

from shapely import LinearRing, LineString, MultiLineString, Point
from shapely.ops import substring

from nexat_trace.planning import curve_calculation
from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.planning.track_graph.primary_track_graph_node import PrimaryTrackGraphNode
from nexat_trace.planning.track_graph.track_graph_node import TrackGraphNode
from nexat_trace.shared import progress
from nexat_trace.shared.config import CurveType, RoutePlanningConfig, CorridorStrategy
from nexat_trace.shared.curve import Curve
from nexat_trace.shared.planning_messages import PlanningMsg
from nexat_trace.track_system import TrackSystem
from nexat_trace.util import geom_tools as gt


class Route:
    """
    Class that holds a path in different forms as well as metadata like turns taken and distance traveled distance on headlands.

    Attributes
    ----------

    turns : Dict
        Dictionary of [CurveType, int] holding count of all turn types present in the route.

    distance_on_headland : float
        Distance in m the vehicle travels on the headlands throughout the route.

    route_error : bool
        Wether or not there is a severe route error present.

    smoothing_error : bool
        Wether or not there is an error in the curve calculation present.

    planning_messages : List[PlanningMsg]
        A list of messages / warnings that occurred during the planning process
    """

    def __init__(
            self,
            track_system: TrackSystem,
            nodes: List[TrackGraphNode],
            metrics: EdgeMetrics,
            target_headlands: List[LinearRing],
            field_border: LinearRing,
            inner_border: LinearRing,
            ab_lines: List[LineString],
            route_params: RoutePlanningConfig,
            is_point_navigation: bool = False,
            progress_out: progress.PlanningProgress | None = None):
        """
        Calculates exact path from route planned by route_planning.
        """
        self._track_system: TrackSystem = track_system
        self._nodes: List[TrackGraphNode] = nodes
        self._path: List[Point] = []
        self._segmented_path: List[List[Point]] = []

        self._metrics: EdgeMetrics = metrics
        self.turns: Dict = None
        self.distance_on_headland: float = None
        self.distance_curves: float = None

        self._target_headlands = target_headlands
        self._field_border = field_border
        self._inner_border = inner_border
        self._ab_lines = ab_lines
        self._line = None
        self._multi_line = None
        self._route_params = route_params.copy()
        self._is_point_navigation = is_point_navigation
        # {ring_index: True/False}
        self._circled_cutouts = None

        self.route_error = False
        self.smoothing_error = False

        self._covered_area = None

        self.planning_messages = []

        if progress_out is None:
            progress_out = progress.PlanningProgress()

        self.progress_out = progress_out

        self.progress_out.planning_stage = progress.PlanningStage.PLANNING_CURVES
        self.progress_out.planning_percent = 75

        double = []
        for i in range(0, len(self._nodes) - 1):
            if self._nodes[i] == self._nodes[i + 1]:
                double.append(i)
        for i in double[::-1]:
            self._nodes.pop(i)

        self._calculate_path()

    def get_linestring(self) -> LineString | None:
        """
        Returns the route path as single LineString including all changes in driving direction.
        """
        return self._line

    def get_multilinestring(self) -> MultiLineString | None:
        """
        Returns the path segmented into single driving directions.

        A segment has a single direction of travel (forwards / backwards) relative to the vehicle. Whenever there is a direction
        change in the route a new segment is used.
        """
        return self._multi_line

    def get_array(self) -> List[Point] | None:
        """
        Returns the route path as a List of Points.
        """
        return self._path

    def get_segmented_array(self) -> List[List[Point]] | None:
        """
        Returns the path segmented into single driving directions.

        Gets the path as a segmented List of Lists of Points. Whenever there is a direction change in the route, a new
        list is started.
        """
        return self._segmented_path

    def get_metrics(self) -> EdgeMetrics | None:
        """
        Returns accumulated instance of EdgeMetrics of the overall route.
        """
        return self._metrics

    def _split_parts(self):
        """
        Splits the route into separate curve segments and returns a list of node lists with the split segments.
        """
        parts: List[List[TrackGraphNode]] = []
        parts.append([self._nodes[0]])
        part_index = 0
        for i in range(1, len(self._nodes) - 1, 1):
            prev_node = self._nodes[i - 1]
            node = self._nodes[i]
            next_node = self._nodes[i + 1]
            part_brake = (
                isinstance(node, PrimaryTrackGraphNode)  # headland reached
                or prev_node == next_node  # direction change
            )

            if part_brake:
                parts[part_index].append(node)
                part_index += 1
                parts.append([node])
            else:
                parts[part_index].append(node)

        parts[part_index].append(self._nodes[-1])

        if (not self._route_params.round_trip_route and len(parts) > 0
                and not self._is_point_navigation):
            # see if start & end include all of their ab lines
            first_part: List[TrackGraphNode] = parts[0]
            first_node: PrimaryTrackGraphNode = first_part[0]
            if isinstance(first_node, PrimaryTrackGraphNode) and first_node.primary_neighbor not in first_part:
                # prepend other node of ab line
                first_part.insert(0, first_node.primary_neighbor)

            last_part: List[TrackGraphNode] = parts[-1]
            last_node: PrimaryTrackGraphNode = last_part[-1]
            if isinstance(last_node, PrimaryTrackGraphNode) and last_node.primary_neighbor not in last_part:
                # append other node of ab line
                last_part.append(last_node.primary_neighbor)

        return parts

    def _calculate_path(self):
        """
        Uses curve calculation module to connect the locations with curves with a radius specified in the route params.
        """
        if self._route_params.debug_prints:
            print("Calculating path")
        smooth_headlands = []
        for headland in self._target_headlands:
            smoothed = gt.erode_linearring(headland, self._route_params.vehicle_turning_radius)
            if smoothed is not None and not smoothed.is_empty:
                smooth_headlands.append(smoothed)
            else:
                smooth_headlands.append(headland)
                if self._route_params.debug_prints:
                    print("erode_linearring did not return valid value")
                if PlanningMsg.CUTOUT_RING_SMOOTHING_ERROR not in self.planning_messages:
                    self.planning_messages.append(PlanningMsg.CUTOUT_RING_SMOOTHING_ERROR)

        self._target_headlands = smooth_headlands

        # setup cutout working flags
        distinct_ring_indexes = []
        self._circled_cutouts = None

        if self._route_params.fully_circle_cutouts:
            self._circled_cutouts = {}
            for node in self._nodes:
                if node.ring_index not in distinct_ring_indexes:
                    distinct_ring_indexes.append(node.ring_index)

            for index in distinct_ring_indexes:
                self._circled_cutouts[index] = False

        self.turns = {}
        for t in CurveType:
            self.turns[t] = 0

        multi_headland = MultiLineString(self._target_headlands)
        self.distance_on_headland = 0.0
        self.distance_curves = 0.0

        mixed_path: List[TrackGraphNode | Point] = []
        self._path = []

        def check_curve_and_extend(curve: Curve):
            if curve.valid:
                self.turns[curve.curve_type] += 1
            else:
                self.smoothing_error = True
                self.turns[CurveType.UNDEFINED] += 1
            points = [Point(coord) for coord in curve.path.coords]
            mixed_path.extend(points)

            buffered_curve = curve.path.buffer(0.01)
            intersection = buffered_curve.intersection(multi_headland)
            if isinstance(intersection, (LineString, MultiLineString)):
                self.distance_on_headland += intersection.length
                self.distance_curves += curve.path.length - intersection.length
            else:
                self.distance_curves += curve.path.length

        parts = self._split_parts()

        for i, part in enumerate(parts):
            p = i / len(parts)
            self.progress_out.planning_percent = 75 + int(p * 24)

            from_node = part[0]
            to_node = part[-1]
            curve = None

            if (isinstance(from_node, PrimaryTrackGraphNode) and
                    isinstance(to_node, PrimaryTrackGraphNode)):
                # plan u turn
                if from_node.primary_neighbor == to_node:
                    mixed_path.extend(part)

                elif to_node.index in from_node.edges and from_node.edges[to_node.index][1].is_neighbor_curve:
                    curve = curve_calculation.trace_neighbor_curve(
                        part,
                        0,
                        -1,
                        self._target_headlands,
                        self._field_border,
                        self._inner_border,
                        self._route_params
                    )
                    check_curve_and_extend(curve)
                else:
                    curve = curve_calculation.connect_ab_lines(
                        part,
                        self._route_params,
                        0,
                        -1,
                        self._target_headlands,
                        self._field_border,
                        self._inner_border,
                        circled_cutouts=self._circled_cutouts
                    )
                    check_curve_and_extend(curve)

            else:
                # start or end of route segment is on the headland
                curve = curve_calculation.trace_curve(
                    part,
                    0,
                    -1,
                    self._target_headlands,
                    self._field_border,
                    self._inner_border,
                    self._route_params
                )
                check_curve_and_extend(curve)

                if isinstance(to_node, PrimaryTrackGraphNode) and curve is not None:
                    # extend path into ab line
                    extension_segment = to_node.get_ab_line()
                    extension_length = min(
                        extension_segment.length,
                        self._route_params.direction_change_extension_distance
                    )
                    extension_point = extension_segment.interpolate(
                        extension_segment.length - extension_length
                    )

                    mixed_path.append(extension_point)

        for i, element in enumerate(mixed_path):

            if isinstance(element, Point):
                self._path.append(element)
            elif i == len(mixed_path) - 1:
                continue
            elif (isinstance(element, PrimaryTrackGraphNode)
                    and isinstance(mixed_path[i + 1], PrimaryTrackGraphNode)):
                # insert ab line (substring) into curve coords
                ab_line: LineString = mixed_path[i + 1].get_ab_line()

                start_projection = 0.0
                if i > 1:
                    start = mixed_path[i - 1]
                    if isinstance(start, TrackGraphNode):
                        start = start.position
                    start_projection = ab_line.project(start)

                end_projection = ab_line.length
                if i < len(mixed_path) - 2:
                    end = mixed_path[i + 2]
                    if isinstance(end, TrackGraphNode):
                        end = end.position
                    end_projection = ab_line.project(end)

                sub_string = substring(
                    ab_line,
                    start_projection,
                    end_projection
                )
                substring_points = [Point(c) for c in sub_string.coords]
                self._path.extend(substring_points)

        if self._route_params.debug_prints:
            print(f"Closing path to loop: {self._route_params.round_trip_route}")
        coords = []
        coords.extend(self._path)
        if self._route_params.round_trip_route:
            coords.append(self._path[0])
        self._line = LineString(coords)

    def _finalize(self):
        """
        Function that is called after route post processing is applied.
        """
        if self._line is None or len(self._line.coords) < 3:
            if self._route_params.debug_prints:
                print("tried to finalize route without valid path line")
            return

        self._metrics.distance = self._line.length
        self._metrics.time = _calculate_drive_time(
            self._line,
            self._route_params.vehicle_speed_straight,
            self._route_params.vehicle_speed_curve,
            self._route_params.delay_on_direction_change
        )

        segments, _ = gt.segment_line(self._line, radius_threshold = self._route_params.vehicle_turning_radius / 10.0)
        self._multi_line = MultiLineString(segments)

        self._segmented_path = []
        for segment in segments:
            self._segmented_path.append(
                [Point(coord) for coord in segment.coords]
            )
        if (
            self._route_params.disable_pi_curves
            and not self._route_params.corridor_strategy == CorridorStrategy.DRIVE_NONE
            and len(self._segmented_path) > 1
        ):
            self._line = LineString()
            self._segmented_path = []
            self._multi_line = MultiLineString()
            self._metrics.distance = 0
            self._metrics.time = 0
            self.planning_messages.append(PlanningMsg.UNEXPECTED_SEGMENTATION)
            if self._route_params.debug_prints:
                print("Got a segmented path, when we shouldn't")
                

        self._calculate_area()

        if self.smoothing_error:
            self.planning_messages.append(PlanningMsg.CURVE_CALCULATION_ERROR)

        self.progress_out.planning_percent = 100
        self.progress_out.planning_stage = progress.PlanningStage.IDLE

        if self._route_params.debug_prints:
            print("Finalized route")

    def _is_valid(self):
        """
        Returns True if no route errors or smoothing errors are detected.
        """
        return not self.route_error and not self.smoothing_error

    def _calculate_area(self):
        """
        Calculates the covered area of the path.
        """
        multi_ab_lines = MultiLineString(self._ab_lines)
        working_area = multi_ab_lines.buffer(
            7.05, cap_style=2, join_style=2, mitre_limit=0.01
        )
        route_coverage = self._line.buffer(
            (self._route_params.working_width / 2) + 0.1,
            cap_style=2,
            join_style=2,
            mitre_limit=0.01
        )
        intersection = route_coverage.intersection(working_area)
        self._covered_area = intersection.area

    def get_covered_area(self) -> float:
        """
        Returns the covered area in m².
        """
        if self._covered_area is None:
            self._calculate_area()
        return self._covered_area


def _calculate_drive_time(
        path_line: LineString,
        speed_straight: float,
        speed_curve: float,
        time_on_direction_change: float = 15.0) -> float:
    """
    Returns the rough estimate of driving time of a given path using the given driving speeds.
    """
    duration = 0.0
    for i in range(len(path_line.coords) - 2):
        p1 = path_line.coords[i]
        p2 = path_line.coords[i + 1]
        l1 = LineString([p1, p2])
        p3 = path_line.coords[i + 2]
        l2 = LineString([p2, p3])
        angle = abs(gt.angle_between_lines(l1, l2))
        if l1.length < 15.0:
            vehicle_speed = speed_curve
        else:
            vehicle_speed = speed_straight

        distance = l1.length
        duration += distance / vehicle_speed

        if angle > pi * 0.9:
            duration += time_on_direction_change

    return duration
