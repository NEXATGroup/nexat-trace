from math import pi
from typing import List, Tuple

from shapely import LinearRing, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import substring
from shapely.strtree import STRtree

from nexat_trace.planning.path import Path
from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.planning.track_graph.primary_track_graph_node import PrimaryTrackGraphNode
from nexat_trace.planning.track_graph.secondary_track_graph_node import SecondaryTrackGraphNode
from nexat_trace.planning.track_graph.track_graph_node import TrackGraphNode
from nexat_trace.shared import progress
from nexat_trace.shared.config import RoutePlanningConfig
from nexat_trace.shared.exceptions import GraphConstructionError
from nexat_trace.shared.weights import Weights
from nexat_trace.track_system import TrackSystem
from nexat_trace.util import geom_tools as gt
from nexat_trace.util.field_conversion import get_ab_lines_on_path, get_headland_index_of_path_on_track_system
from nexat_trace.util.geom_tools import angle_between_lines


class TrackGraph:
    """
    Class that represents a Graph that consists of primary nodes that connect AB-Lines and secondary headland nodes.

    Edges are weighted and connections are only made where a
    curve could be driven by vehicle.
    """

    def __init__(
            self,
            track_system: TrackSystem,
            route_params: RoutePlanningConfig,
            primary_nodes: List[PrimaryTrackGraphNode],
            secondary_nodes: List[SecondaryTrackGraphNode],
            target_headlands: List[LinearRing],
            field_border: LinearRing,
            inner_border: MultiPolygon,
            progress_out: progress.PlanningProgress,
            exclude_areas: List[Polygon] | None = None):

        if route_params.debug_prints:
            print("Building track graph")

        progress_out.planning_stage = progress.PlanningStage.TRACK_GRAPH
        progress_out.planning_percent = 10

        if len(primary_nodes) % 2 != 0:
            raise GraphConstructionError("Sum of primary nodes not devisable by 2")

        self.track_system: TrackSystem = track_system
        self.route_params: RoutePlanningConfig = route_params
        self.target_headlands: List[LinearRing] = target_headlands
        self.primary_nodes: List[PrimaryTrackGraphNode] = primary_nodes
        self.field_border: LinearRing = field_border
        self.inner_border: MultiPolygon = inner_border

        self.exclude_areas: List[Polygon] = []
        if exclude_areas is not None:
            self.exclude_areas = exclude_areas.copy()

        self.all_nodes: List[TrackGraphNode] = primary_nodes.copy()
        self.secondary_nodes: List[SecondaryTrackGraphNode] = secondary_nodes
        self.all_nodes.extend(secondary_nodes)

        self.node_tree = STRtree([n.position for n in self.all_nodes])
        self.node_map = {}

        for node in primary_nodes:
            self.node_map[node.index] = node
        for node in self.secondary_nodes:
            self.node_map[node.index] = node

        self.ab_lines = []
        for i in range(0, len(primary_nodes) - 1, 2):
            node1 = primary_nodes[i]
            node2 = primary_nodes[i + 1]
            node1.link_primary_neighbor(node2, route_params)
            self.ab_lines.append(node1.get_ab_line())

        self.line_tree = STRtree(self.ab_lines)
        self.secondary_node_tree = STRtree([node.position for node in self.secondary_nodes])
        self.primary_node_tree = STRtree([node.position for node in self.primary_nodes])

        for primary in primary_nodes:
            headland = min(self.target_headlands, key = lambda ring: ring.distance(primary.position))
            success = primary.set_secondaries(
                self.secondary_nodes,
                self.route_params,
                self.secondary_node_tree,
                field_border,
                self.inner_border.geoms[0].exterior,  # TODO implement actual support for multiple sub fields
                headland
            )
            if not success:
                primary.plot()
                raise GraphConstructionError("Could not find secondary Nodes for primary")

        progress_out.planning_percent = 15

        # link edges for neighboring primary nodes
        # for direct neighbor curves
        for i, primary in enumerate(primary_nodes):
            primary: PrimaryTrackGraphNode
            for ii in range(i + 1, len(primary_nodes)):
                other = primary_nodes[ii]
                crossing_distance = primary.intersect_secondary.position.distance(
                    other.intersect_secondary.position
                )
                if (primary.distance_to(other) < self.route_params.direct_curve_link_distance
                        and crossing_distance < self.route_params.direct_curve_link_distance
                        and primary.distance_to(other) > 10.0
                        and primary.ring_index == other.ring_index
                        and not route_params.disable_pi_curves):
                    metrics = primary.calculate_metrics(other, route_params)
                    primary.link_node(other, metrics)

        last_known_path_headland_index = None
        missed_path_share = 1 / len(self.route_params.driven_paths) if self.route_params.driven_paths else 0

        for path in self.route_params.driven_paths:
            if self.route_params.debug_prints:
                print("Searching node route by path")

            route = self.get_route_nodes_from_path(path)
            headland_index = get_headland_index_of_path_on_track_system(path, self.track_system)

            if route_params.debug_prints:
                print(f"Found headland index {headland_index} for path")
                print(f"Nodes of the last path: {[node.index for node in route]}")

            if last_known_path_headland_index is not None and headland_index != headland_index:
                if route_params.debug_prints:
                    print(
                        """
                        Provided paths in RoutePlanningConfig.driven_paths have different target headlands.
                        All paths that don't match the target headland of the first path are discarded.
                        """
                    )

                continue

            last_known_path_headland_index = headland_index

            for i in range(1, len(route)):
                node = route[i - 1]
                next_node = route[i]

                def off_ab_penalty(primary_node: PrimaryTrackGraphNode, secondary_node: SecondaryTrackGraphNode):

                    for key in primary_node.edges:

                        other_node = primary_node.edges[key][0]

                        if not isinstance(other_node, SecondaryTrackGraphNode):
                            continue

                        # Penalize other paths off the primary
                        if key != secondary_node.index:
                            primary_node.edges[key][1].missed_path_share += missed_path_share

                        # Penalize other paths onto primary
                        if key != secondary_node.index or not route_params.symmetric_turn_tracks:
                            other_node.edges[primary_node.index][1].missed_path_share += missed_path_share

                def on_ab_penalty(secondary_node: SecondaryTrackGraphNode, primary_node: PrimaryTrackGraphNode):

                    for key in primary_node.edges:

                        other_node = primary_node.edges[key][0]

                        if not isinstance(other_node, SecondaryTrackGraphNode):
                            continue

                        # Penalize other paths onto primary
                        if key != secondary_node.index:
                            other_node.edges[primary_node.index][1].missed_path_share += missed_path_share

                        # Penalize other paths off the primary
                        if key != secondary_node.index or not route_params.symmetric_turn_tracks:
                            primary_node.edges[key][1].missed_path_share += missed_path_share

                # mark metrics as driven track
                if isinstance(node, PrimaryTrackGraphNode) and isinstance(next_node, SecondaryTrackGraphNode):
                    off_ab_penalty(node, next_node)

                elif isinstance(node, SecondaryTrackGraphNode) and isinstance(next_node, PrimaryTrackGraphNode):
                    on_ab_penalty(node, next_node)

        progress_out.planning_percent = 20

        self.latest_block_index = 0
        self._blockify(
            self.route_params.max_block_size,
            self.route_params.min_block_size
        )
        progress_out.planning_percent = 25

        if route_params.debug_prints:
            print("Track graph built")

        if route_params.debug_plot_track_graph:
            from nexat_trace.util import plot_geometry as pg
            self.plot()
            # import os
            # pathString = os.path.expanduser("~/Documents/Debug/fieldGraph.svg")
            # os.makedirs(os.path.dirname(pathString), exist_ok=True)
            # pg.save_fig(pathString)
            pg.show_plot()

    def plot_field(self, headlands):
        """
        Plots field using plot_geometry.
        """
        from nexat_trace.util import plot_geometry as pg
        pg.plot_linestring_list(self.ab_lines, "black", 4)
        pg.plot_linestring(self.field_border, "grey", 4)
        pg.plot_linestring_list(headlands, "black", 4)

    def get_node(self, index) -> TrackGraphNode | None:
        """
        Returns track graph node with given id.
        """
        return self.node_map.get(index, None)

    def get_nearest_node(self, geom) -> TrackGraphNode:
        """
        Returns nearest track graph node to given geometry.
        """
        index = self.node_tree.query_nearest(geom)
        if index is not None:
            return self.all_nodes[index[0]]
        else:
            return None

    def _get_working_width_subset(
            self,
            working_width: float,
            start: Point | int = None) -> List[PrimaryTrackGraphNode]:
        """
        Returns a subset of primary nodes depending on the working_width and optional start.
        """

        # filter ab lines depending on working width
        # number of ab lines that can be left out in between of the ones that have to be traversed
        skip_ab_lines = round(working_width // self.route_params._track_width)
        if skip_ab_lines == 0:
            return self.primary_nodes
        nodes: List[PrimaryTrackGraphNode] = []

        if start is None:
            # build up grid of linestrings that mark the lines that need to be skipped
            subsets = []
            for i in range(skip_ab_lines):
                offset = self.route_params._track_width * i
                subsets.append(self._get_mask_with_offset(skip_ab_lines, offset))

            multi_ab = MultiLineString(self.ab_lines)
            working_area = multi_ab.buffer(
                (self.route_params._track_width / 2.0) + 0.01, cap_style=2, join_style=2, mitre_limit=0.01
            )

            optimal_index = 0
            max_coverage = 0.0
            subset_lines = []
            for i, nodes in enumerate(subsets):
                lines = [node.get_ab_line() for node in nodes]
                subset_lines.append(lines)
                multi_line = MultiLineString(lines)
                subset_coverage = multi_line.buffer(
                    (working_width / 2) + 0.05, cap_style=2, join_style=2, mitre_limit=0.01
                )
                # uncovered_area = working_area.difference(subset_coverage)
                intersection = subset_coverage.intersection(working_area)
                if intersection.area > max_coverage:
                    optimal_index = i
                    max_coverage = intersection.area

            mask = MultiLineString(subset_lines[optimal_index])

        else:
            mask = self._get_mask_with_offset(skip_ab_lines, 0.0, start)
            lines = [node.get_ab_line() for node in mask]
            mask = MultiLineString(lines)

        nodes_indexes = self.primary_node_tree.query(mask, "dwithin", 1.0)
        nodes = [self.primary_nodes[i] for i in nodes_indexes]
        nodes.sort(key=lambda node: node.index)

        return nodes

    def _get_mask_with_offset(
            self,
            skip_ab_lines: int,
            offset: float,
            start: Point | int = None) -> List[PrimaryTrackGraphNode]:
        """
        Returns the subset of ab lines masked by num of skip_ab_lines, offset and optional start.
        """

        if any(len(ab_line.coords) > 2 for ab_line in self.ab_lines):
            return self._get_mask_with_offset_multi_vertex_ab_lines(
                offset,
                start
            )

        keep_lines = []
        start_node = self.primary_nodes[0]
        if start is not None:
            if isinstance(start, Point):
                index = self.primary_node_tree.nearest(start)
                start_node = self.primary_nodes[index]
            elif isinstance(start, int) and abs(start) < len(self.primary_nodes):
                start_node = self.primary_nodes[start]

        coords = []
        coords.append(start_node.position)
        coords.append(start_node.primary_neighbor.position)
        ls = LineString(coords)
        ls = gt.extend_line(ls, 1000000.0)

        def handle_parallel_offset(input_line: LineString, distance: float) -> LineString:
            output = input_line.parallel_offset(distance)
            if isinstance(output, MultiLineString):
                new_coords = []
                for line in output.geoms:
                    new_coords.extend(list(line.coords))
                return LineString(new_coords)

            return output

        if offset > 0.0:
            ls = handle_parallel_offset(ls, offset)
        keep_lines.append(ls)
        last_line = ls
        distance = self.route_params._track_width * skip_ab_lines
        for _ in range(len(self.primary_nodes) // (skip_ab_lines)):
            new_line = handle_parallel_offset(last_line, distance)
            keep_lines.append(new_line)
            last_line = new_line

        last_line = ls
        distance = -distance
        for _ in range(len(self.primary_nodes) // (skip_ab_lines)):
            new_line = handle_parallel_offset(last_line, distance)
            keep_lines.append(new_line)
            last_line = new_line

        multi_keep_lines = MultiLineString(keep_lines)
        indexes = self.primary_node_tree.query(
            multi_keep_lines,
            "dwithin",
            (self.route_params._track_width / 2.0) - 0.01
        )
        subset = [self.primary_nodes[i] for i in indexes]

        return subset

    def _get_mask_with_offset_multi_vertex_ab_lines(
            self,
            offset: float,
            start: Point | int = None) -> List[PrimaryTrackGraphNode]:

        keep_lines = []
        start_node = self.primary_nodes[len(self.primary_nodes) // 2]
        if start is not None:
            if isinstance(start, Point):
                index = self.primary_node_tree.nearest(start)
                start_node = self.primary_nodes[index]
            elif isinstance(start, int) and start in self.node_map:
                start_node = self.primary_nodes[start]

        all_lines: List[LineString] = self.ab_lines.copy()

        start_line = None
        for line in all_lines:
            if line.dwithin(start_node.position, (self.route_params._track_width / 2.0) - 0.1):
                start_line = line
                break

        if offset != 0.0:
            offset_start_line = start_line.parallel_offset(offset, join_style="mitre")
            # TODO ensure that there are still lines "to the right"
            nearest = min(all_lines, key = lambda line: line.distance(offset_start_line))
            start_line = nearest

        all_lines.remove(start_line)
        all_lines.sort(key = lambda line: line.distance(start_line), reverse = True)

        min_distance = self.route_params.working_width - self.route_params._track_width / 2

        keep_lines = [start_line]
        while len(all_lines) > 0:
            multi_keep_lines = MultiLineString(keep_lines)
            line = all_lines.pop()
            inner_buffer = line.buffer(min_distance, 8, "flat", "mitre")

            if multi_keep_lines.intersects(inner_buffer):
                continue

            keep_lines.append(line)

        multi_keep_lines = MultiLineString(keep_lines)
        indexes = self.primary_node_tree.query(
            multi_keep_lines,
            "dwithin",
            (self.route_params._track_width / 2.0) - 0.01
        )
        subset = [self.primary_nodes[i] for i in indexes]

        return subset

    def plot(self, full=True, texts=True):
        """
        Plots track graph using plot_geometry.
        """
        from nexat_trace.util import plot_geometry as pg

        color_edge_ab = "blue"
        color_node_primary = "blue"
        color_edge_primary = "pink"
        color_edge_curve = "green"
        color_edge_secondary = "red"
        color_node_secondary = "red"
        weights = Weights(Weights.GRAPH_BUILDING)
        cnt = 0
        for primary in self.primary_nodes:
            if full and texts:
                index = f"  {primary.index}\n  {primary.block_index}"
            else:
                index = None

            pg.plot_point(primary.position, "0", 10, color_node_primary, index)

            if primary.primary_neighbor is not None:
                line = primary.get_ab_line()
                pg.plot_linestring(line, color=color_edge_ab, linewidth=3, text=None)

            if full:
                for _, (edge_node, metrics) in primary.edges.items():
                    with_arrow = False
                    if isinstance(edge_node, PrimaryTrackGraphNode):
                        if edge_node == primary.primary_neighbor:
                            continue
                        else:
                            pg.plot_linestring(
                                LineString([primary.position, edge_node.position]),
                                color_edge_primary,
                                3,
                                None,
                                with_arrow
                            )
                    else:
                        line = LineString([primary.position, edge_node.position])
                        if (cnt % 5 == 0 or True) and texts:
                            text = f" {metrics.get_cost(weights):.2f}"
                        else:
                            text = None
                        if metrics.working_corridor_error:
                            pg.plot_linestring(line, "purple", 3, text, with_arrow)
                        else:
                            pg.plot_linestring(line, color_edge_curve, 3, text, with_arrow)

                cnt += 1

        for secondary in self.secondary_nodes:
            if full:
                index = f" {secondary.index}\n  {secondary.ring_index}"
            else:
                index = None
            secondary: SecondaryTrackGraphNode
            pg.plot_point(secondary.position, "0", 10, color_node_secondary, index)
            for neighbor, neighbor_metrics in secondary.edges.values():
                line = LineString([secondary.position, neighbor.position])
                cost = secondary.get_metrics(secondary.front_secondary.index).get_cost(weights)
                text = None
                if texts:
                    text = f" {cost:.2f}"
                if isinstance(neighbor, PrimaryTrackGraphNode):
                    if neighbor_metrics.working_corridor_error:
                        pg.plot_linestring(line, "purple", 2, text)
                    else:
                        pg.plot_linestring(line, color_edge_curve, 2, text)
                elif isinstance(neighbor, SecondaryTrackGraphNode):
                    pg.plot_linestring(line, color_edge_secondary, 3, text)
        pg.plot_update()

    def plot_primaries(self):
        """
        Plots primary nodes using plot_geometry.
        """
        from nexat_trace.util import plot_geometry as pg
        for node in self.primary_nodes:
            pg.plot_point(node.position, markersize=12, color="lightgrey", text=f" {node.index}")

    def _blockify(self, max_block_size, min_block_size, buffer_size=13.0):
        """
        Sets the block indexes for primary nodes.
        """

        if max_block_size < 5:
            return

        # get centroids of all ab lines and create a buffer around them
        centroid_buffers = []
        ab_lines = []
        for i in range(0, len(self.primary_nodes) - 1, 2):
            node = self.primary_nodes[i]
            ab_lines.append(self.primary_nodes[i].get_ab_line())
            line = node.get_ab_line()
            centroid = line.centroid
            centroid_buffers.append(centroid.buffer(buffer_size))

        unioned_centroids = gt.union_intersecting_geoms(centroid_buffers)

        line_groups = []

        for buffer in unioned_centroids:
            subset_indexes = self.line_tree.query(buffer, predicate="intersects")
            subset = [ab_lines[index] for index in subset_indexes]
            if len(subset) >= min_block_size:
                line_groups.append(subset)

        for group in line_groups:
            self._check_block_group(group, buffer_size, min_block_size, max_block_size)

    def _check_block_group(self, group, buffer_size, min_block_size, max_block_size):
        """
        Checks and sets block ids for a given group of ab lines.
        """

        end_buffers = []
        end_buffers.extend([Point(line.coords[0]).buffer(buffer_size) for line in group])
        end_buffers.extend([Point(line.coords[-1]).buffer(buffer_size) for line in group])
        unioned_ends = gt.union_intersecting_geoms(end_buffers)

        if len(unioned_ends) < 2:
            return

        if len(unioned_ends) > 2:
            # check all combinations
            unioned_ends.sort(key=lambda poly: 1.0 / poly.exterior.length)
            for i, end_a in enumerate(unioned_ends):
                for ii in range(i, len(unioned_ends)):
                    end_b = unioned_ends[ii]
                    if end_a == end_b:
                        continue
                    multi_buffer = MultiPolygon([end_a, end_b])
                    combination_subset_indexes = self.line_tree.query(multi_buffer, "intersects")
                    candidate_subset = [
                        self.ab_lines[index] for index in combination_subset_indexes
                    ]
                    subset = [
                        c for c in candidate_subset if c.intersects(end_a) and c.intersects(end_b)
                    ]
                    if len(subset) >= min_block_size:

                        self._partition_block_group(subset, max_block_size, min_block_size)
        else:
            subset = group
            if len(subset) >= min_block_size:
                self._partition_block_group(subset, max_block_size, min_block_size)

    def _partition_block_group(self, group, min_block_size, max_block_size):
        """
        Splits group of ab lines into partitions.
        """

        group_len = len(group)
        if group_len < min_block_size:
            return

        test_pos = group[0].parallel_offset(10000000.0, join_style="mitre").centroid
        group.sort(key=lambda line: line.centroid.distance(test_pos))

        partitions = []

        if group_len >= min_block_size and group_len <= max_block_size:
            # make group single block
            partitions = [group]
            self._blockify_partitions(partitions)
            return

        # else look for favorable partitioning size
        partitioning_size = max_block_size
        for divisor in range(max_block_size, max_block_size, -1):
            if group_len % divisor == 0:
                partitioning_size = divisor
                break

        def partition_block(size):
            num_blocks = group_len // size
            start_index = 0
            end_index = size - 1

            for _ in range(num_blocks):
                yield group[start_index:end_index + 1]
                start_index = end_index + 1
                end_index += size

        partitions.extend(list(partition_block(partitioning_size)))

        rest = group_len % partitioning_size
        if rest >= min_block_size:
            partitions.append(
                group[group_len - rest - 1:]
            )

        self._blockify_partitions(partitions)

    def _blockify_partitions(self, partitions):
        """
        Sets block ids for primary nodes in blocks of given partitions.
        """

        for block in partitions:
            for line in block:
                node_indexes = self.primary_node_tree.query(line, "dwithin", 1.0)
                if len(node_indexes) > 2:
                    raise GraphConstructionError(
                        "Error while blockifying: found more than 2 primary nodes on 1 line"
                    )
                self.primary_nodes[node_indexes[0]].block_index = self.latest_block_index
                self.primary_nodes[node_indexes[1]].block_index = self.latest_block_index
            self.latest_block_index += 1

    def search_from_to(
            self,
            from_index,
            to_index,
            weights: Weights | None = None,
            exhaustive = True,
            allow_ab_lines = False,
            illegal_indexes: List[int] | None = None) -> Tuple[List[int], EdgeMetrics]:
        """
        Searches the least costly route between 2 primary nodes on the field graph.

        Parameters
        ----------

        - from_index : int
            Index of the start node
        - to_index : int
            Index of the target node
        - weights : Weights
            Set of cost weights that determine the cost of a route
        - exhaustive : bool
            Continue searching if there already is a route from start to target in net graph
        - allow_ab_lines : bool
            If True will allow paths over ab lines
        - illegal_indexes : List[int]
            List of node indexes that will not be considered

        Returns
        -------

        Tuple of (List[int], EdgeMetrics)
        """

        if weights is None:
            weights = Weights(Weights.ONLY_DISTANCE)

        if illegal_indexes is None:
            illegal_indexes = []

        # Setup Nodes
        start_node = self.get_node(from_index)
        end_node = self.get_node(to_index)

        illegal_nodes = []
        illegal_nodes.extend(illegal_indexes)

        # Initialize pathfinding
        agenda: List[Path] = [Path([start_node], end_node, weights = weights, is_ab = True)]
        target = Path([start_node], end_node, is_ab = True)
        paths = {}

        # Find illegal nodes
        if isinstance(start_node, PrimaryTrackGraphNode):

            illegal_nodes.append(start_node.intersect_secondary.index)

            if not allow_ab_lines:
                illegal_nodes.append(start_node.primary_neighbor.index)

            if start_node.primary_neighbor == end_node:
                return ([from_index, to_index], target.metrics)

        if isinstance(end_node, PrimaryTrackGraphNode):

            illegal_nodes.append(end_node.intersect_secondary.index)

            if not allow_ab_lines:
                illegal_nodes.append(end_node.primary_neighbor.index)

        # Explore Graph
        while agenda and (target.cost == float('inf') or exhaustive):

            if agenda[0].estimated_cost > target.cost:
                break

            current_path = agenda.pop(0)

            # Iterate neighbors
            for index, (neighbor, metrics) in current_path.edges():

                # Goal found?
                if index == to_index:

                    backwards_metrics = neighbor.edges[current_path.last_pos()][1]
                    target = current_path.expand(neighbor, metrics, backwards_metrics)

                    if not exhaustive:
                        break

                # Illegal?
                if isinstance(neighbor, PrimaryTrackGraphNode) and not allow_ab_lines:
                    continue

                if index in illegal_nodes:
                    continue

                # Check new path
                backwards_metrics = neighbor.edges[current_path.last_pos()][1]
                new_path: Path = current_path.expand(neighbor, metrics, backwards_metrics)

                insert = False
                purge = False

                # Path known?
                if index in paths:

                    if new_path.cost < paths[index].cost:
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

                # Remember new path
                if insert:

                    paths.update({new_path.last_pos(): new_path})

                    ids = new_path.indices()
                    if len(ids) >= 3:
                        is_ring_hop = (self.get_node(ids[-3]).ring_index != self.get_node(ids[-2]).ring_index
                                       or self.get_node(ids[-2]).ring_index != self.get_node(ids[-1]).ring_index)
                        if is_ring_hop:
                            line1 = LineString([self.get_node(ids[-3]).position, self.get_node(ids[-2]).position])
                            line2 = LineString([self.get_node(ids[-2]).position, self.get_node(ids[-1]).position])
                            if abs(angle_between_lines(line1, line2)) > pi / 2:
                                if self.route_params.debug_prints:
                                    print(
                                        f"skipped neighbor {to_index} for node {from_index} because of sharp angle in TrackGraph"
                                        + " search"
                                    )
                                continue

                    i = 0
                    while i < len(agenda) and new_path.estimated_cost > agenda[i].estimated_cost:
                        i += 1

                    agenda.insert(i, new_path)

        if target.distance == 0:
            return ([node.index for node in target.path], target.metrics)

        else:
            return ([], target.metrics)

    def _get_secondaries_between_primaries_on_path(
            self,
            primary1: PrimaryTrackGraphNode,
            primary2: PrimaryTrackGraphNode,
            path: LineString) -> List[SecondaryTrackGraphNode]:
        """
        Returns a list of secondary nodes that were traversed on a given path.
        """
        if primary1.primary_neighbor == primary2:
            # that's just an ab line
            return []

        first_intersect_secondary: SecondaryTrackGraphNode = primary1.intersect_secondary
        second_intersect_secondary: SecondaryTrackGraphNode = primary2.intersect_secondary

        path_segment = substring(
            path,
            path.project(
                Point(primary1.get_ab_line().coords[-1])
            ),
            path.project(
                Point(primary2.get_ab_line().coords[-1])
            )
        )

        if not isinstance(path_segment, LineString):
            return []

        if primary1.ring_index == primary2.ring_index:
            return self._get_secondaries_between_secondaries_on_path(
                first_intersect_secondary,
                second_intersect_secondary,
                path_segment
            )
        else:
            if self.route_params.debug_prints:
                print(
                    "Encountered headland hop while searching for nodes included in given path.\n"
                    + "Resulting list of nodes may be incomplete or wrong."
                    + f" at Primary indexes: {primary1.index}, {primary2.index}"
                )

            current_node = first_intersect_secondary
            nodes = []
            projection = 0.0
            while projection < path_segment.length:
                index = self.secondary_node_tree.nearest(path_segment.interpolate(projection))
                projection += 1.0
                nearest_node = self.secondary_nodes[index]
                if nearest_node == current_node:
                    continue
                elif (nearest_node.get_metrics(current_node.index) is None
                      and len(nodes) > 1
                      and nearest_node.get_metrics(nodes[-2].index) is not None):
                    # node is not connected on the graph to the current node
                    # we should keep the one closest to the path not just any that is at any point the closest to the path
                    if nearest_node.position.distance(path_segment) < current_node.position.distance(path_segment):
                        nodes.pop()
                    else:
                        continue
                if nearest_node == second_intersect_secondary:
                    return nodes

                current_node = nearest_node
                nodes.append(current_node)

            return nodes

    def _get_secondaries_between_secondaries_on_path(
            self,
            from_node: SecondaryTrackGraphNode,
            to_node: SecondaryTrackGraphNode,
            path_segment: LineString) -> List[SecondaryTrackGraphNode]:
        """
        Returns the list of secondaries included in the given path on the graph.
        """
        # it could be incorrect to assume that the path always takes the shortest distances on the loaded graph
        # so look for the path that is most near the given path
        def loop_around_secondary_ring(front_direction: bool) -> List[SecondaryTrackGraphNode]:
            candidate_list: List[SecondaryTrackGraphNode] = []

            # run along front / back secondary until target is found
            current_secondary = from_node
            while True:
                if front_direction:
                    next_node = current_secondary.front_secondary
                else:
                    next_node = current_secondary.back_secondary

                if next_node == to_node:
                    return candidate_list

                candidate_list.append(next_node)
                current_secondary = next_node

        front_candidates = loop_around_secondary_ring(True)
        back_candidates = loop_around_secondary_ring(False)

        front_dir_avg_distance = float("inf")
        back_dir_avg_distance = float("inf")

        if len(front_candidates) > 0:
            front_dir_avg_distance = (
                sum(
                    node.position.distance(path_segment) for node in front_candidates
                ) / len(front_candidates)
            )

        if len(back_candidates) > 0:
            back_dir_avg_distance = (
                sum(
                    node.position.distance(path_segment) for node in back_candidates
                ) / len(back_candidates)
            )

        candidate: List[SecondaryTrackGraphNode]
        other_candidate: List[SecondaryTrackGraphNode]

        if front_dir_avg_distance < back_dir_avg_distance:
            candidate = front_candidates
            other_candidate = back_candidates
        else:
            candidate = back_candidates
            other_candidate = front_candidates

        score_difference = abs(front_dir_avg_distance - back_dir_avg_distance)
        if score_difference > 5.0 or len(candidate) < 2 or len(other_candidate) < 2:
            # if the score shows a large difference return the fitting one
            return candidate

        # else check the run of the projection along the path and decide that way
        # this is edge case handling for path segments circling a cutout
        projection_fits = (
            path_segment.project(candidate[0].position) <
            path_segment.project(candidate[1].position)
        )

        if projection_fits:
            return candidate
        else:
            return other_candidate

    def get_route_nodes_from_path(self, path: LineString | MultiLineString) -> List[TrackGraphNode]:
        """
        Returns a list of track graph nodes that are being traversed by the given path on the loaded field.
        """

        if isinstance(path, MultiLineString):
            coords = []
            [coords.extend(segment.coords) for segment in path.geoms]
            path = LineString(coords)
        elif isinstance(path, LineString):
            pass
        else:
            raise TypeError("Path has to be LineString or MultiLineString")

        ab_line_set = get_ab_lines_on_path(self.track_system, path)

        #  sort ab lines along the run of the path
        ab_line_set.sort(key=lambda line: path.project(line.centroid))

        # align the lines with the run of the path
        for i in range(len(ab_line_set)):
            line = ab_line_set[i]
            if path.project(Point(line.coords[-1])) < path.project(Point(line.coords[0])):
                ab_line_set[i] = line.reverse()

        nodes = []

        # cut off path to first ab line
        did_cut_path_to_first_node = False
        cut_path_points_coords = list(path.coords)
        primary_node_multipoint = MultiPoint([node.position for node in self.primary_nodes])
        first_point_index = 0
        while not Point(cut_path_points_coords[first_point_index]).dwithin(primary_node_multipoint, 5.0):
            first_point_index += 1

        if first_point_index > 0:
            did_cut_path_to_first_node = True
            if self.route_params.debug_prints:
                print(f"Cutting path to first primary node, cut off {first_point_index} points")

        cut_path = LineString(cut_path_points_coords[first_point_index::])
        if self.route_params.debug_prints:
            print(f"Cut path to first node: {did_cut_path_to_first_node}, cut off {first_point_index} points")

        start_node = self.primary_nodes[self.primary_node_tree.nearest(Point(cut_path.coords[0]))]
        start_line = ab_line_set[0]
        first_line_index = 0

        if (not did_cut_path_to_first_node
                and start_node.position.distance(start_line.centroid) < Point(cut_path.coords[1]).distance(start_line.centroid)):
            # does route start with a turn to headland or traversal of ab line?
            # start with turn to headland
            first_line_index = 1
            # insert nodes on way to first ab line
            nodes.extend(
                self._get_secondaries_between_primaries_on_path(
                    start_node,
                    self.primary_nodes[self.primary_node_tree.nearest(Point(ab_line_set[1].coords[0]))],
                    cut_path
                )
            )

        i = first_line_index
        while i < len(ab_line_set):
            line = ab_line_set[i]

            node1 = self.primary_nodes[self.primary_node_tree.nearest(Point(line.coords[0]))]
            node2 = self.primary_nodes[self.primary_node_tree.nearest(Point(line.coords[-1]))]

            if node1.distance_to(node2) < 1.0:
                i += 1
                continue

            if node1.primary_neighbor != node2:
                node2 = node1.primary_neighbor
                skip_to_line = min(
                    ab_line_set[i:],
                    key=lambda set_line: set_line.distance(node2.position)
                )
                # should fix infinite loop but may result in broken paths
                skip_index = ab_line_set.index(skip_to_line)
                ab_line_set[i], ab_line_set[skip_index] = ab_line_set[skip_index], ab_line_set[i]

                if self.route_params.debug_prints:
                    print(
                        "Consecutive primary nodes were not part of the same AB line in the loaded graph - "
                        + "Skipping to line of primary neighbor"
                    )
                    print(f"Swapped from ab_ index {i} to {skip_index} while searching for nodes on path")

            nodes.append(node1)
            nodes.append(node2)

            if i < len(ab_line_set) - 1:
                # search secondary nodes to next ab line

                next_line = ab_line_set[i + 1]
                next_primary = self.primary_nodes[self.primary_node_tree.nearest(Point(next_line.coords[0]))]
                while next_primary == node2 and i < len(ab_line_set) - 2:
                    i += 1
                    next_line = ab_line_set[i + 1]
                    next_primary = self.primary_nodes[self.primary_node_tree.nearest(Point(next_line.coords[0]))]
                    if next_primary == node2 and i == len(ab_line_set) - 2 and self.route_params.debug_prints:
                        print(
                            "Could not find next primary node on path after current ab line - "
                            + "Skipping search for secondary nodes to next ab line"
                        )

                nodes.extend(self._get_secondaries_between_primaries_on_path(node2, next_primary, cut_path))

            i += 1

        # TODO roundtrip path?

        return nodes
