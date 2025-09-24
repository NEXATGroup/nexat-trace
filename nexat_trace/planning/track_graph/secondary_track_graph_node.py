from shapely import LineString, Point

from nexat_trace.planning.track_graph.track_graph_node import TrackGraphNode
from nexat_trace.shared.config import RoutePlanningConfig


class SecondaryTrackGraphNode(TrackGraphNode):
    """
    Class that represents a Graph node in a Graph where there are primary and secondary nodes with only certain allowed edges.

    The edges between these nodes can actually be driven by the vehicle.
    """

    def __init__(self, position: Point, ring_index):
        super().__init__(position, ring_index)
        self.front_secondary: SecondaryTrackGraphNode = None
        self.back_secondary: SecondaryTrackGraphNode = None
        self.intersect_primary = None

    def link_back(self, node: TrackGraphNode, route_params: RoutePlanningConfig):
        """
        Links back secondary with metrics calculated with route params.

        Also links self to other nodes front.
        """
        if self.back_secondary == node:
            return
        self.back_secondary = node
        metrics = self.calculate_metrics(node, route_params)
        node.link_front(self, route_params)
        self.link_node(node, metrics)

    def link_front(self, node: TrackGraphNode, route_params: RoutePlanningConfig):
        """
        Links front secondary with metrics calculated with route params.

        Also links self to other nodes back.
        """
        if self.front_secondary == node:
            return
        metrics = self.calculate_metrics(node, route_params)
        self.front_secondary = node
        node.link_back(self, route_params)
        self.link_node(node, metrics)

    def check_drivability(
            self,
            primary: TrackGraphNode,
            secondary_neighbor: TrackGraphNode,
            route_params: RoutePlanningConfig | None = None) -> bool:
        """
        Checks if a curve from given primary to secondary neighbor is possible.
        """

        l1 = LineString([self.position, secondary_neighbor.position])
        if secondary_neighbor == self.front_secondary:
            l2 = LineString(
                [secondary_neighbor.position, secondary_neighbor.front_secondary.position]
            )
        elif secondary_neighbor == self.back_secondary:
            l2 = LineString(
                [secondary_neighbor.position, secondary_neighbor.back_secondary.position]
            )
        else:
            return False

        ab_line: LineString = primary.get_ab_line()

        # if heuristic_corridor_angle == 0 (default value): Perform exact working corridor error calculation
        if not 0 < route_params.weights.heuristic_corridor_angle < 1:

            safe_zone_line = LineString(
                [
                    ab_line.interpolate(0.4, True),
                    ab_line.interpolate(0.6, True)
                ]
            )
            safe_zone = safe_zone_line.buffer(route_params.vehicle_turning_radius * 2, cap_style = "flat", join_style = "mitre")

            return not (l1.intersects(safe_zone) or l2.intersects(safe_zone))

        return True
