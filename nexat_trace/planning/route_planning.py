import threading
import time
from typing import List, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from shapely import LinearRing, LineString, MultiPolygon, Point

from nexat_trace.planning.net_graph.net_graph import NetGraph
from nexat_trace.planning.route import Route
from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.planning.track_graph.primary_track_graph_node import (
    PrimaryTrackGraphNode,
)
from nexat_trace.planning.track_graph.secondary_track_graph_node import (
    SecondaryTrackGraphNode,
)
from nexat_trace.planning.track_graph.track_graph import TrackGraph
from nexat_trace.planning.track_graph.track_graph_node import TrackGraphNode
from nexat_trace.shared import progress
from nexat_trace.shared.config import RoutePlanningConfig
from nexat_trace.shared.exceptions import RoutePlanningError
from nexat_trace.shared.planning_messages import PlanningMsg
from nexat_trace.shared.weights import Weights
from nexat_trace.track_system import TrackSystem
from nexat_trace.util import field_conversion
from nexat_trace.util import geom_tools as gt

"""
This module can be used to define and solve vehicle routing problems using
ortools solvers and track / net graphs. Finds the shortest paths over a
given track graph
"""


def net_graph_from_track_system(
        track_system: TrackSystem,
        route_params: RoutePlanningConfig,
        progress_out: progress.PlanningProgress,
        limit: int = None) -> tuple[
            NetGraph, List[LineString], LinearRing, LinearRing | None, List[PlanningMsg]]:
    """
    Builds a net graph from field data json.

    Parameters
    ----------
    limit
        max iterations each step the way finding will take during graph construction

    Returns
    -------

        Tuple of (net_graph, ab_lines, headland_shape, outer_field_border)
    """
    TrackGraphNode.reset_indexes()

    tw = field_conversion.get_track_width(track_system)
    route_params._track_width = tw
    if route_params.debug_prints:
        print(f"Track width is {route_params._track_width}m")

    ab_lines, secondaries, planning_msgs = field_conversion.track_system_to_primaries_secondaries(
        track_system,
        route_params
    )

    headlands, _ = field_conversion.get_target_headland_from_track_system(
        track_system,
        route_params
    )
    outer_headland_shape = headlands[0]
    outer_field_border = track_system.outer_border.exterior
    border = track_system.outer_border
    inner_border = field_conversion.get_inner_border_from_track_system(track_system, route_params)

    primary_nodes: List[PrimaryTrackGraphNode] = []
    ab_line: LineString
    for ab_line in ab_lines:
        n1 = PrimaryTrackGraphNode(Point(ab_line.coords[0]), None)
        n1.set_ab_line(ab_line)
        primary_nodes.append(n1)
        n2 = PrimaryTrackGraphNode(Point(ab_line.coords[-1]), None)
        n2.set_ab_line(ab_line)
        primary_nodes.append(n2)
    ab_lines = []
    for i in range(0, len(primary_nodes) - 1, 2):
        ls = primary_nodes[i].get_ab_line()
        ab_lines.append(ls)

    track_system.ab_lines = ab_lines
    
    if route_params.debug_plot_field:
        from nexat_trace.util import plot_geometry as pg
        pg.plot_linestring(outer_field_border, "black")
        for poly in list(inner_border.geoms):
            pg.plot_linestring(poly.exterior, "lightgrey")
        pg.plot_linestring_list(headlands, "grey")
        pg.plot_linestring_list(ab_lines, "grey")

    track_graph = TrackGraph(
        track_system,
        route_params,
        primary_nodes,
        secondaries,
        headlands,
        border,
        inner_border,
        progress_out
    )

    # build network graph
    net = NetGraph(
        track_graph,
        route_params,
        progress_out,
        limit
    )

    return net, ab_lines, outer_headland_shape, outer_field_border, planning_msgs


def get_route(routing,
              route_params: RoutePlanningConfig,
              solution,
              track_graph: TrackGraph,
              net_graph: NetGraph,
              ab_lines,
              field_border,
              progress_out: progress.PlanningProgress) -> Tuple[Route, List[PlanningMsg]]:
    """
    Constructs a Route object from the routing solution.
    """
    warnings = []

    index_from = routing.Start(0)
    locations_path = []
    locations_path.append(index_from)
    route_objective_sum = 0

    indexes = []

    index = routing.Start(0)
    while not routing.IsEnd(index):
        indexes.append(index)
        index = solution.Value(routing.NextVar(index))

    route_metrics = EdgeMetrics()
    for i in range(0, len(indexes) - 1):
        from_index = indexes[i]
        to_index = indexes[i + 1]

        cost = net_graph.get_cost(from_index, to_index, route_params)

        route_metrics = EdgeMetrics.add(route_metrics, net_graph.get_metrics(from_index, to_index))

        locations_path.append(to_index)

        route_objective_sum += cost

    cost = net_graph.get_cost(indexes[-1], indexes[0], route_params)
    locations_path.append(indexes[0])

    # filter out virtual locations not bound to actual track graph nodes
    locations_path = [loc for loc in locations_path if net_graph.get_track_node_location(loc) is not None]

    route_nodes = []

    route_error = not is_route_valid(locations_path)

    if route_params.round_trip_route:
        route_range = range(len(locations_path) - 1)
    else:
        route_range = range(len(locations_path) - 2)

    for path_step in route_range:

        from_node_index = locations_path[path_step]
        to_node_index = locations_path[path_step + 1]

        nodes_indexes, _ = net_graph.get_path_location_indexes(from_node_index, to_node_index)
        if len(nodes_indexes) == 0:
            warnings.append(PlanningMsg.NON_DRIVABLE_ROUTE)
            print("Route error: not all paths drivable by vehicle")
            print(f"Error on route segment {from_node_index} -> {to_node_index}")
        else:
            for i in range(0, len(nodes_indexes) - 1):
                node_index = nodes_indexes[i]
                route_nodes.append(net_graph.get_track_node(node_index))
            node_index = nodes_indexes[-1]
            route_nodes.append(net_graph.get_track_node(node_index))

    if route_params.debug_prints:
        print(f"Total route objective value: {route_objective_sum}")
        # print route node indices
        print([n.index for n in route_nodes])

    inner_border = None
    if isinstance(track_graph.inner_border, MultiPolygon):
        # TODO implement actual multi inner border support
        inner_border = list(track_graph.inner_border.geoms)[0].exterior
    else:
        raise TypeError("Inner border in track graph was not MultiPolygon")

    route = Route(
        track_graph.track_system,
        route_nodes,
        route_metrics,
        track_graph.target_headlands,
        field_border,
        inner_border,
        ab_lines,
        route_params,
        progress_out = progress_out
    )
    route._route_error = route_error

    return route, warnings


def is_route_valid(indexes) -> bool:
    """
    Checks a list of location indexes for errors in the route.

    Returns True if no error is found.
    """
    valid = True
    location_ring = indexes.copy()
    location_ring.pop(-1)

    check_range = range(0, len(indexes) - 1, 2)
    if abs(indexes[0] - indexes[1]) != 1:
        check_range = range(-1, len(indexes) - 2, 2)

    try:
        for i in check_range:
            frm = location_ring[i]
            to = location_ring[i + 1]
            if abs(frm - to) != 1:
                valid = False
    except IndexError:
        return valid

    return valid


def solve_route(
        net_graph: NetGraph,
        route_params: RoutePlanningConfig,
        progress_out: progress.PlanningProgress,
        ab_lines,
        headland_shape: LinearRing,
        field_border: LinearRing,
        solver_time_limit = -1) -> Tuple[Route | None, List[PlanningMsg] | None]:
    """
    Uses the current state of the config module to search a route over the given net graph.

    Parameters
    ----------

    - net_graph : NetGraph
        Instance of fully connected net graph of the field
    - solver_time_limit : int
        <= 0: greedy descent search method, > 0: searching time in seconds with guided local search
    """
    if route_params.debug_prints:
        print("Searching route")

    warnings = []

    if route_params._track_width is None:
        tw = field_conversion.get_track_width(net_graph.track_graph.track_system)
        route_params._track_width = tw

    starting_location: int | None = None
    starting_node: PrimaryTrackGraphNode | None = None
    finish_location: int | None = None
    finish_node: PrimaryTrackGraphNode | None = None

    starting_point = route_params.starting_point
    finish_point = route_params.finish_point

    if starting_point is None:
        # no starting point given
        # start top left corner (when in utm)
        x, _, __, y = field_border.bounds
        starting_point = Point((x, y))

    nodes: List[PrimaryTrackGraphNode] = net_graph.track_graph.primary_nodes.copy()

    if (isinstance(starting_point, int)
            and abs(starting_point) < len(net_graph.track_graph.primary_nodes)):
        starting_point = net_graph.track_graph.primary_nodes[starting_point].position

    if isinstance(starting_point, Point):
        # search node nearest to the starting point

        # sort nodes by distance to starting point
        nodes.sort(key=lambda node: starting_point.distance(node.position))
        # start looking for nodes in working subset nearest to
        # the starting point for start location
        looking_index = 0
        while starting_location is None and looking_index < len(nodes):
            starting_node = nodes[looking_index]

            for location_index, node in net_graph.locations.items():
                if node.index == starting_node.index:
                    starting_location = location_index
                    break
            looking_index += 1

    if not route_params.round_trip_route and finish_point is None:
        # look for nodes in working subset farthest to the starting point for finish location
        looking_index = len(nodes) - 1
        while finish_location is None and looking_index > -1:
            finish_node = nodes[looking_index]

            for location_index, node in net_graph.locations.items():
                if node.index == finish_node.index:
                    finish_location = location_index
                    break
            looking_index -= 1
    elif finish_point is not None:
        if (isinstance(finish_point, int)
                and abs(finish_point) < len(net_graph.track_graph.primary_nodes)):
            finish_point = net_graph.track_graph.primary_nodes[finish_point].position

        if isinstance(finish_point, Point):
            # sort nodes by distance to end point
            nodes.sort(key=lambda node: finish_point.distance(node.position))
            # start looking for nodes in working subset nearest
            # to the finish point for end location
            looking_index = 0
            while finish_location is None and looking_index < len(nodes):
                finish_node = nodes[looking_index]

                for location_index, node in net_graph.locations.items():
                    if node.index == finish_node.index:
                        finish_location = location_index
                        break
                looking_index += 1

    if finish_node is not None and finish_node.primary_neighbor == starting_node:
        # start & end cannot be on the same ab line
        first_end_node_index = finish_node.index
        first_end_location = finish_location
        warnings.append(PlanningMsg.CHANGED_END_POINT)
        location_node_indexes = [node.index for node in net_graph.locations.values()]
        nodes.sort(key=lambda node: finish_point.distance(node.position))
        for i in range(1, len(nodes)):
            new_finish_node_candidate = nodes[i]
            if new_finish_node_candidate == starting_node:
                continue
            if new_finish_node_candidate == finish_node:
                continue

            if new_finish_node_candidate.index in location_node_indexes:
                finish_node = new_finish_node_candidate
                for location_index, node in net_graph.locations.items():
                    if node.index == finish_node.index:
                        finish_location = location_index
                        break
                break
        if route_params.debug_prints:
            nodes_msg = f"Changed finish node from {first_end_node_index} to {finish_node.index}"
            locations_msg = f"(locations {first_end_location} -> {finish_location})"
            print(nodes_msg + " " + locations_msg)

    if starting_location is None:
        raise RoutePlanningError("did not find starting location node")

    if finish_location is None and not route_params.round_trip_route:
        raise RoutePlanningError("did not find finish location node")

    if route_params.debug_prints:
        print(f"Starting location: {starting_location}")
        print(f"Finish location: {finish_location}")

    problem_data = {}
    problem_data["locations"] = len(net_graph.locations.keys())
    problem_data["num_vehicles"] = 1
    problem_data["depot"] = starting_location
    manager = pywrapcp.RoutingIndexManager(
        problem_data["locations"],
        problem_data["num_vehicles"],
        problem_data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    headland_shape = gt.erode_linearring(headland_shape, route_params.vehicle_turning_radius)
    net_graph.cost_cache.clear()

    def distance_callback(from_index, to_index):
        """Returns the calculated cost between the two nodes."""
        from_index = manager.IndexToNode(from_index)
        to_index = manager.IndexToNode(to_index)

        cost = net_graph.get_cost(from_index, to_index, route_params)
        return cost

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    start_and_end_differ = starting_location != finish_location

    if not route_params.round_trip_route and not start_and_end_differ:
        # cannot set up finish location constraint if start & end locations are the same
        route_params.round_trip_route = True
        warnings.append(PlanningMsg.ROUTE_CHANGED_TO_ROUND_TRIP)

    elif not route_params.round_trip_route and start_and_end_differ:

        def time_callback(_from_index, _to_index):
            """
            Returns 1 as the time spent driving from one location to the other.

            That way the finish node can get a time window constraint to be visited last.
            """
            return 1

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        max_time_window = len(nodes) * 4  # set max time to big number > expected max "time" spent
        # Add time windows constraints.
        routing.AddDimension(
            time_callback_index,
            0,  # no slack
            max_time_window,
            True,  # start at zero
            "Time"
        )

        time_dimension = routing.GetDimensionOrDie("Time")
        # Set time window for each location
        # allow broad time windows for intermediate nodes
        for location in net_graph.locations:
            if location == finish_location:
                continue
            index = manager.NodeToIndex(location)
            time_dimension.CumulVar(index).SetRange(0, max_time_window)  # Broad time window

        time_to_reach_finish = len(net_graph.locations.keys()) - 1
        # set time window for finish node to window at end of route
        finish_index = manager.NodeToIndex(finish_location)
        time_dimension.CumulVar(finish_index).SetRange(
            time_to_reach_finish,
            max_time_window
        )

    # heuristic parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if solver_time_limit > 0:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = solver_time_limit
    else:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
        )

    progress_out.planning_stage = progress.PlanningStage.PLANNING_ROUTE

    if solver_time_limit > 0:
        updater = threading.Thread(target=update_progress, args=[solver_time_limit, progress_out])
        updater.start()

    solution = routing.SolveWithParameters(search_parameters)

    if solution is not None:
        if route_params.debug_prints:
            print("Route found")
        route, new_warnings = get_route(
            routing,
            route_params,
            solution,
            net_graph.track_graph,
            net_graph,
            ab_lines,
            field_border,
            progress_out
        )
        warnings.extend(new_warnings)
    else:
        print("Route planning did not find a route")
        return None, []

    return route, warnings


def update_progress(seconds, progress_out: progress.PlanningProgress):
    """
    Updates progress while vrp search time is running.
    """

    for s in range(seconds):
        progress_out.planning_percent = 50 + int((s / seconds) * 25)
        time.sleep(1.0)


def navigate_from_to(
        route_params: RoutePlanningConfig,
        start_geom,
        target_geom,
        net_graph: NetGraph,
        fallback = False) -> Route | None:
    """
    Returns a new Route object leading from start geom to target geom.

    Start and target geoms can be Point or LineString to get desired heading.
    Will insert a turn maneuver in route if headings do not match.
    """

    if not isinstance(start_geom, Point) and not isinstance(start_geom, LineString):
        raise TypeError("Start geometry has to be Point on end of ab line")
    if not isinstance(target_geom, Point) and not isinstance(target_geom, LineString):
        raise TypeError("Start and target geometries have to be Point or LineString")

    track_graph = net_graph.track_graph
    # TODO implement actual multi inner border support
    inner_border = list(track_graph.inner_border.geoms)[0].exterior

    start_heading_point = None
    target_heading_point = None
    if isinstance(start_geom, LineString):
        start_node = track_graph.get_nearest_node(Point(start_geom.coords[0]))
        start_heading_point = Point(start_geom.coords[-1])

    elif isinstance(start_geom, Point):
        start_node = track_graph.get_nearest_node(start_geom)

    if isinstance(target_geom, LineString):
        target_node = track_graph.get_nearest_node(Point(target_geom.coords[0]))
        target_heading_point = Point(target_geom.coords[-1])

    elif isinstance(target_geom, Point):
        target_node = track_graph.get_nearest_node(target_geom)

    if isinstance(start_node, PrimaryTrackGraphNode) and isinstance(target_node, PrimaryTrackGraphNode):
        # use net graph shortest ways
        route_indexes, metrics = net_graph.get_path(
            start_node.index,
            target_node.index,
            fallback
        )
        route_nodes = [net_graph.get_track_node(i) for i in route_indexes]

        route = Route(
            net_graph.track_graph.track_system,
            route_nodes,
            metrics,
            track_graph.target_headlands,
            track_graph.field_border,
            inner_border,
            track_graph.ab_lines,
            route_params,
            True
        )
        return route

    if isinstance(start_node, SecondaryTrackGraphNode) or isinstance(target_node, SecondaryTrackGraphNode):
        disable_indexes = []
        if isinstance(start_node, PrimaryTrackGraphNode):
            disable_indexes.append(start_node.primary_neighbor.index)
        elif isinstance(target_node, PrimaryTrackGraphNode):
            disable_indexes.append(target_node.primary_neighbor.index)
        if start_heading_point is not None:
            disable_indexes.extend(
                get_disable_indexes_for_heading(
                    start_node
                )
            )
        if target_heading_point is not None:
            disable_indexes.extend(
                get_disable_indexes_for_heading(
                    target_node
                )
            )

        weights = Weights(Weights.ONLY_DISTANCE)
        route_indexes_1, metrics_1 = track_graph.search_from_to(
            start_node.index,
            target_node.index,
            weights=weights,
            exhaustive=True,
            allow_ab_lines=True,
            illegal_indexes=disable_indexes
        )
        route_indexes_2, metrics_2 = track_graph.search_from_to(
            start_node.index,
            target_node.index,
            weights=weights,
            exhaustive=True,
            allow_ab_lines=False,
            illegal_indexes=disable_indexes
        )
        if route_indexes_1 is None and route_indexes_2 is None:
            if route_params.debug_prints:
                print("navigate_from_to(): both comparation routes were None")
            return None

        if metrics_1.get_cost(weights) < metrics_2.get_cost(weights):
            route_indexes = route_indexes_1
            metrics = metrics_1
        else:
            route_indexes = route_indexes_2
            metrics = metrics_2
        if route_indexes is None:
            if route_params.debug_prints:
                print("navigate_from_to(): both compared route was None")
            return None

        route_nodes = [net_graph.get_track_node(i) for i in route_indexes]
        # check if turn needed
        start_forward = False
        if isinstance(route_nodes[0], PrimaryTrackGraphNode):
            start_forward = True
        else:
            if (route_nodes[0].position.distance(start_heading_point) >
                    route_nodes[1].position.distance(start_heading_point)):
                start_forward = True

        target_forward = False
        if isinstance(route_nodes[-1], PrimaryTrackGraphNode):
            target_forward = True
        else:
            if (route_nodes[-2].position.distance(target_heading_point) >
                    route_nodes[-1].position.distance(target_heading_point)):
                target_forward = True

        needs_turn = start_forward != target_forward

        if needs_turn:
            if isinstance(target_node, SecondaryTrackGraphNode):
                # search for a turn before target headland node
                last_primary_index = -1
                for i in range(-2, -len(route_nodes) + 1, -1):
                    if isinstance(route_nodes[i], PrimaryTrackGraphNode):
                        last_primary_index = i
                        break
                if last_primary_index != -1 and last_primary_index < len(route_nodes) - 1:
                    middle_node = route_nodes[last_primary_index].intersect_secondary
                    if (route_nodes[last_primary_index + 1] ==
                            middle_node.front_secondary):
                        turn_other_way_node = middle_node.back_secondary
                        turn_extension_node = turn_other_way_node.back_secondary
                    else:
                        turn_other_way_node = middle_node.front_secondary
                        turn_extension_node = turn_other_way_node.front_secondary

                    # insert turn in other direction
                    route_nodes.insert(last_primary_index + 1, turn_other_way_node)
                    route_nodes.insert(last_primary_index + 1, turn_extension_node)
                    route_nodes.insert(last_primary_index + 1, turn_other_way_node)
                    route_nodes.insert(last_primary_index + 1, middle_node)

                else:
                    # no turn before target headland node found
                    if len(route_nodes) < 6:
                        raise RoutePlanningError("Route was not long enough to insert turn")
                    # insert turn maneuver before target node
                    turn_inserted = False
                    for i in range(-3, -len(route_nodes) + 3, -1):
                        if (isinstance(route_nodes[i], SecondaryTrackGraphNode)
                                and isinstance(route_nodes[i - 1], SecondaryTrackGraphNode)
                                and isinstance(route_nodes[i - 2], SecondaryTrackGraphNode)):
                            route_nodes[i - 1] = route_nodes[i - 1].intersect_primary
                            turn_inserted = True
                            break
                    if not turn_inserted:
                        raise RoutePlanningError("Route was not long enough to insert turn")
            else:
                # drive further and insert turn behind target
                intersect_target = target_node.intersect_secondary
                if not isinstance(route_nodes[-2], SecondaryTrackGraphNode):
                    raise RoutePlanningError("Could not insert turn into route")
                extension_nodes = []
                # get direction
                if intersect_target.front_secondary == route_nodes[-2]:
                    extension_nodes = [
                        intersect_target,
                        intersect_target.back_secondary,
                        intersect_target.back_secondary.back_secondary,
                        intersect_target.back_secondary
                    ]
                else:
                    extension_nodes = [
                        intersect_target,
                        intersect_target.front_secondary,
                        intersect_target.front_secondary.front_secondary,
                        intersect_target.front_secondary
                    ]
                # insert extension to route
                route_nodes.pop(-1)
                route_nodes.extend(extension_nodes)
                route_nodes.append(target_node)

        route = Route(
            net_graph.track_graph.track_system,
            route_nodes,
            metrics,
            track_graph.target_headlands,
            track_graph.field_border,
            inner_border,
            track_graph.ab_lines,
            route_params,
            True
        )

    return route


def get_disable_indexes_for_heading(
        node: SecondaryTrackGraphNode) -> List[int]:
    """
    Returns the indexes of neighboring primary nodes to disable wayfinding to make room for turning.
    """
    if node is None:
        return []

    disable_range = 1

    disabled_nodes = []
    if node.intersect_primary is not None:
        disabled_nodes.append(node.intersect_primary)
    current_node = node
    for _ in range(disable_range):
        current_node = current_node.back_secondary
        disabled_nodes.append(current_node.intersect_primary)

    current_node = node
    for _ in range(disable_range):
        current_node = current_node.front_secondary
        disabled_nodes.append(current_node.intersect_primary)

    # disable path searching for that node
    return [node.index for node in disabled_nodes if node is not None]
