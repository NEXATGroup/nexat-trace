from typing import Dict, Tuple

from shapely import LineString, Point
from shapely.geometry.base import BaseGeometry

from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.shared.config import RoutePlanningConfig
from nexat_trace.util import geom_tools as gt
from nexat_trace.util import plot_geometry as pg


class TrackGraphNode:
    """
    Base class for all TrackGraph nodes.

    Supports weighted directional edges.
    """

    latest_index = 0

    def reset_indexes():
        """
        Resets indexes for graph building.
        """
        TrackGraphNode.latest_index = 0

    def __init__(self, position: Point, ring_index):
        self.index = TrackGraphNode.latest_index
        TrackGraphNode.latest_index += 1
        self.position = position
        self.ring_index = ring_index

        # dict of {index: [node instance, metrics]}
        self.edges: Dict[int, Tuple[TrackGraphNode, EdgeMetrics]] = {}

    def link_node(
            self,
            node,
            metrics: EdgeMetrics | None,
            reverse_metrics: EdgeMetrics | None = None):
        """
        Links self to other with given metrics.

        If the traversal of the same edge in the other direction has other metrics, set
        'reverse_metrics'. Else both directions will get the same traversal parameters.
        """
        if node.index in self.edges:
            return
        self.edges[node.index] = [node, metrics]

        if reverse_metrics is None and metrics is not None:
            node.link_node(self, metrics.copy(), metrics)
        else:
            node.link_node(self, reverse_metrics, metrics)

    def remove_link(self, other):
        """
        Removes link between nodes.
        """
        if other.index not in self.edges:
            return
        self.edges.pop(other.index)
        other.remove_link(self)

    def get_metrics(self, target_index: int) -> EdgeMetrics | None:
        """
        Returns metrics to node with given index if linked.
        """
        if target_index in self.edges:
            return self.edges[target_index][1]
        return None

    def __eq__(self, other):
        """
        Returns True if other is of type TrackGraphNode and indexes match.
        """
        return isinstance(other, TrackGraphNode) and self.index == other.index

    def distance_to(self, other):
        """
        Returns distance between node positions.
        """
        if isinstance(other, BaseGeometry):
            return self.position.distance(other)
        elif isinstance(other, TrackGraphNode):
            return self.position.distance(other.position)

    def calculate_metrics(
            self,
            other,
            route_params: RoutePlanningConfig,
            line1: LineString | None = None,
            line2: LineString | None = None):
        """
        Calculates metrics from self to other track graph node.

        Uses given route params and optional line segments. 'line1' and 'line2' should represent the AB line and the headland
        segment if applicable.
        """
        metrics = EdgeMetrics()
        metrics.distance = self.distance_to(other)
        if self.ring_index != 0 and other.ring_index != 0:
            metrics.distance *= 2
        if line1 is not None and line2 is not None:
            metrics.angle = abs(gt.angle_between_lines(line1, line2))

        # determine the speed
        nexat_speed = get_nexat_speed_on_angle(metrics.angle, route_params)
        metrics.time = metrics.distance / nexat_speed

        return metrics

    def plot(self, color="green", marker="0", markersize=15):
        """
        Plots node using plot_geometry.
        """
        pg.plot_point(self.position, marker, markersize, color)


def get_nexat_speed_on_angle(angle: float, route_params: RoutePlanningConfig):
    """
    Returns approximate nexat speed on given curve angle in radians.
    """
    if not isinstance(angle, float):
        pass
    angle = abs(angle)
    nexat_speed = route_params.vehicle_speed_curve
    if angle < route_params.speed_curve_angle_threshold:
        nexat_speed += (
            (route_params.speed_curve_angle_threshold - angle) / route_params.speed_curve_angle_threshold
        ) * (route_params.vehicle_speed_straight - route_params.vehicle_speed_curve)
    return nexat_speed
