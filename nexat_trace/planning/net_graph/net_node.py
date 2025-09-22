from typing import Dict, Tuple

from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.shared.weights import Weights


class NetNode:
    """
    Class that represents a Graph node in a Graph where all nodes have an edge to every other node.
    """

    def __init__(self, index, block_index = None):
        # dict of {node index: [node instance, metrics]}
        self.edges: Dict[int, Tuple[NetNode, EdgeMetrics]] = {}
        self.index = index
        self.block_index = block_index

    def link_edge(self, node, metrics: EdgeMetrics, reverse_metrics: EdgeMetrics | None = None):
        """
        Saves edge between other node and self with given metrics.

        If the traversal of the same edge in the other direction has other metrics, set
        'reverse_metrics'. Else both directions will get the same traversal parameters.
        """
        weights = Weights(Weights.GRAPH_BUILDING)
        if (self.get_metrics(node.index) is not None
                and self.get_metrics(node.index).get_cost(weights) <= metrics.get_cost(weights)):
            # do not link new edge if there already is an edge to this node with lower cost
            return
        self.edges[node.index] = [node, metrics]
        if reverse_metrics is None and metrics is not None:
            node.link_edge(self, metrics.copy(), metrics)
        else:
            node.link_edge(self, reverse_metrics, metrics)

    def get_metrics(self, target_index) -> EdgeMetrics | None:
        """
        Returns metrics between self and given node index if linked.
        """
        if target_index in self.edges:
            return self.edges[target_index][1]
        return None

    def __eq__(self, other):
        """
        Returns True if other is of type NetNode and indexes match.
        """
        return isinstance(other, NetNode) and self.index == other.index
