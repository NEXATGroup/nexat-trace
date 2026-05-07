from collections import defaultdict
from math import pi
from typing import Dict, List, Tuple

from shapely import LineString, MultiPolygon, Polygon

from nexat_trace.planning.net_graph.net_node import NetNode
from nexat_trace.planning.path import Path
from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.planning.track_graph.primary_track_graph_node import PrimaryTrackGraphNode
from nexat_trace.planning.track_graph.track_graph import TrackGraph
from nexat_trace.planning.track_graph.track_graph_node import TrackGraphNode
from nexat_trace.shared import progress
from nexat_trace.shared.config import RoutePlanningConfig
from nexat_trace.shared.weights import Weights
from nexat_trace.util.geom_tools import angle_between_lines


class NetGraph:
    """
    Class that represents a fully connected Graph of NETNodes with weighted edges.
    """

    def __init__(
            self,
            track_graph: TrackGraph,
            route_params: RoutePlanningConfig,
            progress_out: progress.PlanningProgress,
            limit = 100,
            exclude_areas: List[Polygon] | None = None):

        if route_params.debug_prints:
            print("Building net graph")

        self.nodes: Dict[int, NetNode] = {}  # dict of {node index: [node instance, metrics]}
        self.shortest_ways: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
        self.locations: Dict[int, NetNode] = {}
        self.cost_cache: Dict[int, Dict[int, int]] = defaultdict(dict)

        self.track_graph = track_graph

        self.exclude_areas = []
        if exclude_areas is not None:
            self.exclude_areas = exclude_areas.copy()

        self.progress = progress_out
        self.progress.planning_stage = progress.PlanningStage.NET_GRAPH

        self.working_subset_track_nodes: List[PrimaryTrackGraphNode]
        self.set_subset_mask(route_params)

        # Find shortest ways
        subset_len = len(self.working_subset_track_nodes)
        for i, p_node in enumerate(self.working_subset_track_nodes):

            self.progress.planning_percent = 25 + int((i / subset_len) * 25)
            self.find_neighbors(
                p_node.index,
                [node.index for node in self.working_subset_track_nodes],
                Weights(Weights.GRAPH_BUILDING),
                True,
                min(limit, subset_len - 1)
            )

        # Set edges for ab lines
        index = -1
        for node in track_graph.primary_nodes:
            other = node.primary_neighbor

            if node.index < other.index:
                metrics = node.get_metrics(other.index).copy()
                metrics.distance /= 2

                new_virtual_node = NetNode(index, node.block_index)
                self.nodes[index] = new_virtual_node

                self.set_edge(
                    node.index,
                    index,
                    metrics,
                    metrics.copy(),
                    [node.index, index]
                )
                self.set_edge(
                    other.index,
                    index,
                    metrics.copy(),
                    metrics.copy(),
                    [other.index, index]
                )
                index -= 1

        self.set_exclude_areas(self.exclude_areas)

        if route_params.debug_prints:
            if route_params.debug_plot_net_graph:
                self.plot()
            print("Net graph built")

    def set_edge(
            self,
            from_node_index,
            to_node_index,
            metrics: EdgeMetrics,
            backwards_metrics: EdgeMetrics,
            path: List[int]):
        """
        Sets new edge if distance is lower than the distance already registered for the node pair.
        """
        if from_node_index in self.nodes:
            node = self.nodes[from_node_index]
        else:
            from_block_index = self.track_graph.get_node(from_node_index).block_index
            node = NetNode(from_node_index, from_block_index)
            self.nodes[from_node_index] = node

        if to_node_index in self.nodes:
            other_node = self.nodes[to_node_index]
        else:
            to_block_index = self.track_graph.get_node(to_node_index).block_index
            other_node = NetNode(to_node_index, to_block_index)
            self.nodes[to_node_index] = other_node

        if (node.get_metrics(to_node_index) is not None
                and node.get_metrics(to_node_index).distance <= metrics.distance):
            # do not link new edge if there already is an edge to this node with shorter distance
            return

        node.link_edge(other_node, metrics, backwards_metrics)
        self.shortest_ways[from_node_index].update({to_node_index: path})
        self.shortest_ways[to_node_index].update({from_node_index: path[::-1]})

    def get_cost(
            self,
            from_node_index_location: int,
            to_node_index_location: int,
            route_params: RoutePlanningConfig):
        """
        Returns the currently registered cost between the two location nodes.
        """

        if self.cost_cache[from_node_index_location].get(to_node_index_location) is not None:
            return self.cost_cache[from_node_index_location][to_node_index_location]

        cost: int = 0
        try:
            from_node: NetNode = self.locations[from_node_index_location]
            to_node: NetNode = self.locations[to_node_index_location]

            metrics = from_node.edges[to_node.index][1]
            one_time_offset = 0.0

            if (from_node.block_index != to_node.block_index
                    and from_node.block_index is not None
                    and to_node.block_index is not None):

                one_time_offset = route_params.weights.different_block_penalty

            cost = int(metrics.get_cost(route_params.weights) + one_time_offset)

        except (KeyError, OverflowError):
            cost = 2147483647  # int32 max value

        self.cost_cache[from_node_index_location][to_node_index_location] = cost

        return cost

    def get_path(
            self,
            from_index,
            to_index,
            fallback = False,
            allow_ab_lines = False) -> Tuple[List[int], EdgeMetrics | None]:
        """
        Returns shortest way between given indexes if already found.

        Else will return ([], None)
        """
        # Fallback if from_index or to_index are unknown
        if fallback and to_index not in self.shortest_ways[from_index]:
            return self.track_graph.search_from_to(from_index, to_index, allow_ab_lines=allow_ab_lines)

        from_node = self.get_track_node(from_index)
        to_node = self.get_track_node(to_index)
        if (isinstance(from_node, PrimaryTrackGraphNode) and isinstance(to_node, PrimaryTrackGraphNode)
                and from_node.primary_neighbor == to_node):
            return [from_index, to_index], from_node.edges[to_index][1]

        try:
            indexes = self.shortest_ways[from_index][to_index]
        except KeyError:
            print(f"there was no shortest way found from {from_index} to {to_index}", flush=True)
            return [], None

        return indexes, self.nodes[from_index].get_metrics(to_index)

    def get_path_location_indexes(
            self,
            from_location_index,
            to_location_index) -> Tuple[List[int], EdgeMetrics | None]:
        """
        Translates location indexes to node indexes and returns self.get_path between those indexes.

        Will return ([], None) if indexes not found.
        """
        if from_location_index not in self.locations:
            print(f"from index {from_location_index} not found in location indexes")
            return [], None
        if to_location_index not in self.locations:
            print(f"to index {to_location_index} not found in location indexes")
            return [], None

        from_index = self.locations[from_location_index].index
        to_index = self.locations[to_location_index].index
        return self.get_path(from_index, to_index)

    def get_track_node(self, index) -> TrackGraphNode:
        """
        Returns the node of the track graph with given id.
        """
        return self.track_graph.get_node(index)

    def get_track_node_location(self, location_index) -> TrackGraphNode:
        """
        Translates location index to node index and returns self.get_track_node of that index.
        """
        return self.track_graph.get_node(self.locations[location_index].index)

    def get_metrics(self, from_location, to_location) -> EdgeMetrics:
        """
        Returns metrics between locations.
        """

        to_index = self.locations[to_location].index
        metrics = self.locations[from_location].get_metrics(to_index)

        return metrics

    def set_exclude_areas(self, polys):
        """
        Sets areas to exclude from working subset.
        """
        self.working_subset_indexes.clear()
        multi_poly = MultiPolygon(polys)
        node: PrimaryTrackGraphNode
        for node in self.working_subset_track_nodes:
            ab_line = node.get_ab_line()

            if not multi_poly.intersects(ab_line):
                self.working_subset_indexes.append(node.index)

        new_indices = []
        for index in self.working_subset_indexes:
            # reinsert virtual ab line middle locations
            net_graph_node: NetNode = self.nodes[index]
            for node_index, instance_metrics_tuple in net_graph_node.edges.items():
                if node_index < 0:
                    virtual_node_index = instance_metrics_tuple[0].index
                    if virtual_node_index not in self.working_subset_indexes and virtual_node_index not in new_indices:
                        new_indices.append(virtual_node_index)

        self.working_subset_indexes.extend(new_indices)

        self.fill_locations()

    def fill_locations(self):
        """
        Fills location dict with the working subset.
        """
        self.locations = {}
        if len(self.nodes.items()) == 0:
            return
        for cnt, index in enumerate(self.working_subset_indexes):
            node = self.nodes[index]
            self.locations[cnt] = node

    def set_subset_mask(self, route_params: RoutePlanningConfig):
        """
        Sets the working subset depending on working width and optional starting point.
        """
        if len(route_params.driven_paths) == 0:
            self.working_subset_track_nodes = self.track_graph._get_working_width_subset(
                route_params.working_width,
                route_params.working_mask_start
            )
        else:
            self.working_subset_track_nodes = self.track_graph._get_working_width_subset_from_paths(
                route_params.driven_paths,
                route_params.working_width,
                route_params.working_mask_start
            )
        self.working_subset_indexes = [node.index for node in self.working_subset_track_nodes]
        self.fill_locations()

    def get_location_index_of_track_node(self, node: TrackGraphNode):
        """
        Returns the location index of a track graph node.
        """
        for index, other_node in self.locations.items():
            if other_node.index == node.index:
                return index

    def plot(self):
        """
        Plots net graph using plot_geometry.
        """
        from shapely import LineString  # noqa: I001
        from ...util import plot_geometry as pg

        nodes = list(self.nodes.values())
        for i, node in enumerate(nodes):
            track_node = self.get_track_node(node.index)
            if track_node is None:
                continue
            pg.plot_point(track_node.position)
            for other_node in nodes[i + 1:]:
                other_track_node = self.get_track_node(other_node.index)
                if other_track_node is None:
                    continue
                ls = LineString([track_node.position, other_track_node.position])
                pg.plot_linestring(ls, linewidth=1)
        pg.show_plot()

    def find_neighbors(
            self,
            from_index: int,
            targets: List[int],
            weights: Weights | None = None,
            exhaustive: bool = True,
            limit: int = 20,
            illegal_indexes: List[int] | None = None) -> None:
        """
        Find n nearest neighbors and links them in the net graph.
        """

        if weights is None:
            weights = Weights(Weights.GRAPH_BUILDING)

        if illegal_indexes is None:
            illegal_indexes = []

        # Setup
        found_cnt = 0
        start_node = self.track_graph.get_node(from_index)
        paths = {from_index: Path([start_node], weights=weights, is_ab = True)}

        # find illegal nodes
        illegal_nodes = []
        illegal_nodes.extend(illegal_indexes)

        if isinstance(start_node, PrimaryTrackGraphNode):
            illegal_nodes.append(start_node.intersect_secondary.index)
            illegal_nodes.append(start_node.primary_neighbor.index)

        agenda = [value for value in paths.values()]

        # Explore Graph
        while agenda and found_cnt < limit:

            current_path = agenda.pop(0)

            # Iterate neighbors
            for to_index, (neighbor, metrics) in current_path.edges():

                # Eligible Node?
                if to_index in illegal_nodes:
                    continue

                # Check new path
                backwards_metrics = neighbor.edges[current_path.last_pos()][1]
                new_path: Path = current_path.expand(neighbor, metrics, backwards_metrics, True)
                insert = False
                purge = False

                # Path known?
                if to_index in paths:

                    if new_path.cost < paths[to_index].cost:
                        insert = True
                        purge = True

                else:
                    insert = True

                # Cleanup agenda
                if purge:

                    i = 0
                    while i < len(agenda):

                        if new_path.last_pos() in agenda[i].path:
                            agenda.pop(i)
                        else:
                            i += 1

                    for path in self.shortest_ways[from_index].values():

                        if new_path.last_pos() in path:
                            found_cnt -= 1

                # Remember new path
                if insert:

                    paths.update({new_path.last_pos(): new_path})

                    # Neighbor found?
                    if to_index in targets:
                        ids = new_path.indices()

                        if self.track_graph.get_node(to_index).intersect_secondary.index not in ids:
                            found_cnt += 1
                            self.set_edge(from_index, to_index, new_path.metrics, new_path.backwards_metrics, ids)

                    else:

                        if isinstance(neighbor, PrimaryTrackGraphNode):
                            continue

                        ids = new_path.indices()
                        if len(ids) >= 3:
                            is_ring_hop = (
                                self.track_graph.get_node(ids[-3]).ring_index != self.track_graph.get_node(ids[-2]).ring_index
                                or self.track_graph.get_node(ids[-2]).ring_index != self.track_graph.get_node(ids[-1]).ring_index
                            )
                            if is_ring_hop:
                                line1 = LineString(
                                    [self.track_graph.get_node(ids[-3]).position, self.track_graph.get_node(ids[-2]).position]
                                )
                                line2 = LineString(
                                    [self.track_graph.get_node(ids[-2]).position, self.track_graph.get_node(ids[-1]).position]
                                )
                                if abs(angle_between_lines(line1, line2)) > pi / 2:
                                    if self.track_graph.route_params.debug_prints:
                                        print(f"skipped neighbor {to_index} for node {from_index} because of sharp angle")
                                    continue

                        i = len(agenda) - 1

                        while exhaustive and i > 0 and new_path.cost < agenda[i].cost:
                            i -= 1

                        agenda.insert(i, new_path)
