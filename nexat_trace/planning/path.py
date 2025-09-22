from typing import List, Tuple

from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.planning.track_graph.primary_track_graph_node import PrimaryTrackGraphNode
from nexat_trace.planning.track_graph.track_graph_node import TrackGraphNode
from nexat_trace.shared.weights import Weights


class Path:
    """
    Wrapper class for shortest Path information.
    """

    def __init__(self,
                 path: List[TrackGraphNode],
                 target: TrackGraphNode = None,
                 weights: Weights = None,
                 metrics: EdgeMetrics = None,
                 backwards_metrics: EdgeMetrics = None,
                 is_ab: bool = False):

        self.path = path
        self.target = target
        self.metrics = metrics or EdgeMetrics()
        self.backwards_metrics = backwards_metrics or EdgeMetrics()
        self.metrics.is_ab = is_ab
        self.backwards_metrics.is_ab = is_ab
        self.weights = weights
        self.cost = self.metrics.get_cost(weights) if weights else float("inf")
        self.backwards_cost = self.backwards_metrics.get_cost(weights) if weights else float("inf")
        self.calculate_distance()

    def calculate_distance(self):
        """
        Updates members with distance & cost to self.target.
        """

        if self.target:

            self.distance = self.path[-1].distance_to(self.target)

            if self.weights:
                self.distance = self.distance * self.weights.headland_distance_factor
                self.distance = self.distance ** self.weights.headland_cost_exponent

            self.estimated_cost = self.cost + self.distance

        else:
            self.distance = float('inf')
            self.estimated_cost = float('inf')

    def update_endpoint(self, target: TrackGraphNode) -> None:
        """
        Updates members with a new target.
        """

        self.target = target
        self.calculate_distance()

    def last_pos(self) -> int:
        """
        Returns the last position of the path.
        """
        return self.path[-1].index

    def edges(self) -> List[Tuple[int, Tuple[TrackGraphNode, EdgeMetrics]]]:
        """
        Returns the graph edge entries of the last path item.
        """
        return self.path[-1].edges.items()

    def indices(self) -> List[int]:
        """
        Returns the node indices in the path.
        """
        return [node.index for node in self.path]

    def expand(self,
               node: TrackGraphNode,
               metrics: EdgeMetrics,
               backwards_metrics: EdgeMetrics | None = None,
               normalize = False):
        """
        Appends node to the path with the given traversal metrics.
        """

        if backwards_metrics is None:
            backwards_metrics = metrics.copy()

        # Normalize Path costs between primary and secondary nodes
        last_node = self.path[-1]
        if normalize:

            metrics = metrics.copy()
            backwards_metrics = backwards_metrics.copy()

            if isinstance(last_node, PrimaryTrackGraphNode) and not isinstance(node, PrimaryTrackGraphNode):
                metrics.distance = metrics.distance - last_node.distance_to_headland
                backwards_metrics.distance = backwards_metrics.distance - last_node.distance_to_headland

            if not isinstance(last_node, PrimaryTrackGraphNode) and isinstance(node, PrimaryTrackGraphNode):
                metrics.distance = metrics.distance - node.distance_to_headland
                backwards_metrics.distance = backwards_metrics.distance - node.distance_to_headland

        return Path(
            self.path + [node],
            self.target,
            self.weights,
            EdgeMetrics.add(self.metrics, metrics),
            EdgeMetrics.add(self.backwards_metrics, backwards_metrics)
        )
