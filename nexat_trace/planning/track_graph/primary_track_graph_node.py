from math import pi
from typing import List

from shapely import LinearRing, LineString, Point, Polygon, STRtree

from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.planning.track_graph.track_graph_node import TrackGraphNode
from nexat_trace.shared.config import RoutePlanningConfig
from nexat_trace.shared.weights import Weights
from nexat_trace.util import geom_tools as gt
from nexat_trace.util.field_conversion import get_corridor_line


class PrimaryTrackGraphNode(TrackGraphNode):
    """
    Class that represents a Graph node in a Graph where there are primary and secondary nodes with only certain allowed edges.

    The edges between these nodes can actually be driven by the vehicle
    """

    def __init__(self, position: Point, ring_index):
        super().__init__(position, ring_index)
        self.primary_neighbor: TrackGraphNode | None = None
        self.block_index = None
        self.ab_line: LineString = None

        # that node is not to be crossed if a way is being found to this (self) node
        self.intersect_secondary = None

        # Cost of shortest secondary to headland. Used for cost calculation in net graph.
        self.distance_to_headland = float("inf")

    def link_primary_neighbor(self, other, route_params: RoutePlanningConfig):
        """
        Links the primary neighbor forming an ab line.

        Also links other primary with self.
        """
        if self.primary_neighbor == other:
            return
        self.primary_neighbor = other
        metrics = EdgeMetrics()

        metrics.distance = self.distance_to(other)
        metrics.time = metrics.distance / route_params.vehicle_speed_straight
        metrics.is_ab = True
        self.link_node(other, metrics)
        other.link_primary_neighbor(self, route_params)

    def _link_secondary(self, node: TrackGraphNode, metrics: EdgeMetrics, reverse_metrics: EdgeMetrics | None = None):
        """
        Links secondary node.
        """
        if metrics.distance < self.distance_to_headland:
            self.distance_to_headland = metrics.distance

        self.link_node(node, metrics, reverse_metrics)

    def set_secondaries(
            self,
            secondary_nodes,
            route_params: RoutePlanningConfig,
            secondary_node_tree: STRtree,
            field_border: Polygon,
            inner_border: LinearRing | None = None,
            headland: LinearRing | None = None,
            planner_config: RoutePlanningConfig | None = None):
        """
        Links the 2 neighbor nodes of the intersecting headland node.
        """
        line1 = self.get_ab_line()

        intersecting_secondary = self.search_intersecting_secondary(
            secondary_nodes,
            secondary_node_tree,
            field_border
        )
        self.intersect_secondary = intersecting_secondary
        intersecting_secondary.intersect_primary = self

        self.ring_index = self.intersect_secondary.ring_index

        got_one_secondary = False

        def handle_linking_secondary(secondary: TrackGraphNode, line2: LineString, line3: LineString):
            metrics = self.calculate_metrics(
                secondary,
                route_params,
                line1,
                line2,
                line3
            )
            if inner_border is not None and not 0 < route_params.heuristic_corridor_angle < 1:
                self.check_working_corridor_for_metrics(
                    secondary,
                    metrics,
                    line1,
                    inner_border,
                    headland,
                    planner_config
                )
            reverse_metrics = metrics.copy()

            self._link_secondary(
                secondary,
                metrics,
                reverse_metrics
            )

        secondary = intersecting_secondary.back_secondary
        if intersecting_secondary.check_drivability(self, secondary, route_params):
            line2 = LineString([secondary.position, secondary.back_secondary.position])
            line3 = LineString([intersecting_secondary.front_secondary.position, secondary.position])
            got_one_secondary = True

        secondary = intersecting_secondary.front_secondary
        if intersecting_secondary.check_drivability(self, secondary, route_params):
            line2 = LineString([secondary.position, secondary.front_secondary.position])
            line3 = LineString([intersecting_secondary.back_secondary.position, secondary.position])
            got_one_secondary = True

        if not got_one_secondary:
            secondary1 = intersecting_secondary.back_secondary
            line2 = LineString([secondary1.position, secondary1.back_secondary.position])
            line3 = LineString([intersecting_secondary.front_secondary.position, secondary.position])
            m1 = self.calculate_metrics(secondary1, route_params, line1, line2, line3)
            secondary2 = intersecting_secondary.front_secondary
            line2 = LineString([secondary2.position, secondary.front_secondary.position])
            line3 = LineString([intersecting_secondary.back_secondary.position, secondary.position])
            m2 = self.calculate_metrics(secondary2, route_params, line1, line2, line3)
            weights = Weights(Weights.GRAPH_BUILDING)
            if m1.get_cost(weights) < m2.get_cost(weights):
                self._link_secondary(secondary1, m1)
            else:
                self._link_secondary(secondary2, m2)
            got_one_secondary = True

        return got_one_secondary

    def check_working_corridor_for_metrics(
            self,
            target_node: TrackGraphNode,
            metrics: EdgeMetrics,
            ab_line: LineString,
            inner_border: LinearRing,
            headland_ring: LinearRing,
            route_params: RoutePlanningConfig):
        """
        Checks working corridor risk for traversal to target node.
        """
        # get working corridor geometry
        corridor_line = get_corridor_line(ab_line, route_params.working_width, inner_border)

        # find turn curve radius middle point
        buffer_radius = route_params.vehicle_turning_radius - 0.01
        front_l: Point = corridor_line.parallel_offset(buffer_radius, "left").interpolate(1.0, True)
        front_r: Point = corridor_line.parallel_offset(buffer_radius, "right").interpolate(1.0, True)

        nearest_turn_offset = front_l
        if front_r.distance(target_node.position) < front_l.distance(target_node.position):
            nearest_turn_offset = front_r

        # construct turning path collider

        turn_collider = nearest_turn_offset.buffer(buffer_radius)

        if headland_ring.intersects(turn_collider):
            error_value = (buffer_radius - nearest_turn_offset.distance(headland_ring)) / (buffer_radius * 0.8)
            if corridor_line.distance(headland_ring) < buffer_radius:
                # if target line is placed extra bad, displace sinking value of distance on inner side of turning radius
                error_value += (buffer_radius - corridor_line.distance(headland_ring)) / buffer_radius

            metrics.working_corridor_error = error_value

    def search_intersecting_secondary(
            self,
            secondary_nodes: List[TrackGraphNode],
            secondary_node_tree: STRtree,
            field_border: Polygon) -> TrackGraphNode:
        """
        Searches and returns the intersecting secondary node on headland.
        """
        ab_line = self.get_ab_line()
        ab_line = gt.extend_line_in_bounds(ab_line, field_border.exterior, extend_back = False)
        indexes = secondary_node_tree.query(ab_line, "dwithin", 2.0)
        if len(indexes) == 1:
            return secondary_nodes[indexes[0]]
        elif len(indexes) == 0:
            # return closest node on headland
            sorted_secondaries = secondary_nodes.copy()
            sorted_secondaries.sort(key = lambda node: self.position.distance(node.position))
            return secondary_nodes[0]
        else:
            candidates = [secondary_nodes[i] for i in indexes]
            candidates.sort(key = lambda node: self.position.distance(node.position))
            return candidates[0]

    def get_ab_line(self) -> LineString | None:
        """
        Returns a LineString built from self.primary_neighbor.position => self.position.
        """
        if self.ab_line is not None:
            return self.ab_line
        elif self.primary_neighbor is not None:
            return LineString(
                [self.primary_neighbor.position, self.position]
            )
        return None

    def set_ab_line(self, line: LineString) -> None:
        """
        Sets the ab line member and aligns it the correct way.
        """
        if Point(line.coords[0]).distance(self.position) > Point(line.coords[-1]).distance(self.position):
            self.ab_line = LineString(line)
        else:
            self.ab_line = line.reverse()

    def calculate_metrics(
            self,
            other: TrackGraphNode,
            route_params: RoutePlanningConfig,
            line1 = None,
            line2 = None,
            line3 = None) -> EdgeMetrics:
        """
        Calculates metrics from self to other track graph node.

        Uses given route params and optional line segments. 'line1' and 'line2' should represent the AB line and the headland
        segment if applicable.
        """

        if line1 is not None and line2 is not None:
            metrics.angle = abs(gt.angle_between_lines(line1, line2) / pi)

        # if heuristic_corridor_angle > 0: Estimate working corridor error by angle heuristic
        theta = route_params.heuristic_corridor_angle

        if metrics.angle > 0.0 and 0 < theta < 1:

            angle = abs(gt.angle_between_lines(line1, line3) / pi)

            if (metrics.angle > theta or angle > theta):

                # normalize corridor error to [0,1]
                a = (angle - theta) / (1 - theta)
                b = (metrics.angle - theta) / (1 - theta)

                # Combined error margin
                metrics.working_corridor_error = max(0, (a + b - abs(a * b))) ** 0.5

        metrics = super().calculate_metrics(other, route_params)

        if isinstance(other, PrimaryTrackGraphNode) and other != self.primary_neighbor:
            metrics.is_neighbor_curve = True
            # distance penalty
            metrics.distance *= route_params.neighbor_curve_distance_multiplier
            # distance of pi turn maneuver
            metrics.distance += pi * route_params.vehicle_turning_radius  # 2 * 1/4 circle circumference
            metrics.distance += route_params._track_width * 2.0
            metrics.distance += route_params.direction_change_extension_distance * 2.0

        return metrics
