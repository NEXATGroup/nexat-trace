import math
from math import pi
from typing import Dict, List

from shapely import LinearRing, LineString, MultiLineString, MultiPoint, Point, Polygon, STRtree
from shapely.ops import nearest_points

from nexat_trace.planning.track_graph.edge_metrics import EdgeMetrics
from nexat_trace.planning.track_graph.primary_track_graph_node import PrimaryTrackGraphNode
from nexat_trace.planning.track_graph.secondary_track_graph_node import SecondaryTrackGraphNode
from nexat_trace.shared.config import CorridorStrategy, RoutePlanningConfig
from nexat_trace.shared.curve import Curve, CurveType
from nexat_trace.util import geom_tools as gt
from nexat_trace.util.field_conversion import get_corridor_line

"""
This module is used to compute dubins paths connecting track segments.
"""


def connect_ab_lines(
        nodes,
        route_params: RoutePlanningConfig,
        from_index: int,
        to_index: int,
        headlands: List[LinearRing],
        field_border: LinearRing,
        inner_field_border: LinearRing,
        allow_headland_hop = True,
        circled_cutouts: Dict[int, bool] = None) -> Curve:
    """
    Uses the relations of track graph nodes to trace a curve between nodes over a headland shape.

    Start and end nodes have to be ends of an ab line.
    Returns the dubins path from the starting node to the target node
    and wether or not the u turn path is valid.
    """
    if to_index >= len(nodes):
        raise KeyError("connect_ab_lines() was called with inappropriate indexes")

    from_node = nodes[from_index]
    to_node = nodes[to_index]
    if not isinstance(from_node, PrimaryTrackGraphNode):
        raise TypeError(
            "connect_ab_lines() was called with inappropriate nodes (from_node != PrimaryTrackGraphNode)"
        )
    if not isinstance(to_node, PrimaryTrackGraphNode):
        raise TypeError(
            "connect_ab_lines() was called with inappropriate nodes (to_node != PrimaryTrackGraphNode)"
        )

    if to_node.index in from_node.edges and from_node.edges[to_node.index][1].is_neighbor_curve:
        return trace_neighbor_curve(
            nodes,
            from_index,
            to_index,
            headlands,
            field_border,
            inner_field_border,
            route_params
        )

    over_node = nodes[from_index + 1]

    if not isinstance(over_node, SecondaryTrackGraphNode):
        return Curve([from_node.position, to_node.position], CurveType.UNDEFINED, False)

    if allow_headland_hop and from_node.ring_index != to_node.ring_index and len(nodes) > 3:

        return trace_headland_hop(
            nodes,
            route_params,
            from_index,
            to_index,
            headlands,
            field_border,
            inner_field_border,
            circled_cutouts
        )

    sorted_headlands = headlands.copy()
    sorted_headlands.sort(key=lambda headland: over_node.position.distance(headland))
    headland_shape: LinearRing = sorted_headlands[0]

    # gather locations and generate directed segments to traverse

    from_segment = from_node.get_ab_line()

    segment_tangent_start = over_node.position
    if isinstance(nodes[from_index + 2], PrimaryTrackGraphNode):
        over_extend_node = nodes[from_index + 2].intersect_secondary
    else:
        over_extend_node = nodes[from_index + 2]
    segment_tangent_end = over_extend_node.position

    # check direction of headland against the segment tangent
    headland_with_origin = gt.ring_with_origin_at(
        headland_shape,
        headland_shape.interpolate(headland_shape.project(segment_tangent_start))
    )
    if _is_long_way_around(headland_with_origin, segment_tangent_end):
        # if run of headland shape is other way reverse it
        headland_shape = headland_shape.reverse()

    to_segment = to_node.get_ab_line().reverse()

    headland_segment = gt.get_substring_on_linearring(
        headland_shape,
        from_node.intersect_secondary.position,
        to_node.intersect_secondary.position
    )
    if isinstance(headland_segment, Point):
        # if this happens just do a roundtrip
        headland_segment = gt.ring_with_origin_at(headland_shape, from_node.intersect_secondary.position)
        headland_segment = gt.substring(
            headland_segment,
            0,
            headland_segment.length
        )

    headland_segment_compare = gt.get_substring_on_linearring(
        headland_shape.reverse(),
        from_node.intersect_secondary.position,
        to_node.intersect_secondary.position
    )

    if headland_segment.length > headland_segment_compare.length:
        headland_segment = headland_segment_compare
        headland_shape = headland_shape.reverse()

    # generate curves

    curve1 = None
    curve2 = None

    is_at_cutout = over_node.ring_index != 0

    curve1 = search_curve_to_headland(
        from_segment,
        headland_segment,
        headland_shape,
        field_border,
        inner_field_border,
        is_at_cutout,
        route_params
    )

    curve2 = search_curve_to_ab(
        to_segment,
        headland_segment,
        headland_shape,
        field_border,
        inner_field_border,
        is_at_cutout,
        route_params
    )

    valid = True

    circle_cutout = False
    # check if cutout present
    if to_node.ring_index != 0:
        # check if cutout should be circled
        intersects_outer = headlands[0].intersects(headland_shape)
        if not intersects_outer:
            ring_index = to_node.ring_index
            if circled_cutouts is not None and not circled_cutouts[ring_index]:
                circle_cutout = True
                circled_cutouts[ring_index] = True

    if curve1.valid and curve2.valid:
        # path is completely valid
        # do cutout circling if needed and combine coords
        headland_path_between = gt.get_substring_on_linearring(
            headland_shape, Point(curve1.path.coords[-1]), Point(curve2.path.coords[0])
        )
        if circle_cutout:
            continue_around = gt.ring_with_origin_at(
                headland_shape,
                Point(curve2.path.coords[0])
            )
            circled_coords = (
                list(headland_path_between.coords) +
                list(continue_around.coords)
            )
            headland_path_between = LineString(circled_coords)

        threshold = 1.95 if circle_cutout else 0.95

        if headland_path_between.length > headland_shape.length * threshold:
            # curve1 ends after the start of curve2
            headland_segment = gt.get_substring_on_linearring(
                headland_shape,
                from_node.intersect_secondary.position,
                Point(curve2.path.coords[0])
            )
            if circle_cutout:
                continue_around = gt.ring_with_origin_at(
                    headland_shape,
                    Point(curve2.path.coords[0])
                )
                circled_coords = (
                    list(headland_segment.coords) +
                    list(continue_around.coords)
                )
                headland_segment = LineString(circled_coords)

            combined_coords = (
                list(headland_segment.coords[0:-1])
                + list(curve2.path.coords)
            )
            valid = False
        else:

            if headland_path_between.length > 1.0:
                combined_coords = (
                    list(curve1.path.coords)
                    + list(headland_path_between.coords[1:-1])
                    + list(curve2.path.coords)
                )
            else:
                combined_coords = (
                    list(curve1.path.coords)
                    + list(curve2.path.coords)
                )

    elif not curve1.valid:
        # part 1 not valid
        # save part 2, circle cutouts if needed and combine coords
        valid = False
        headland_segment = gt.get_substring_on_linearring(
            headland_shape,
            from_node.intersect_secondary.position,
            Point(curve2.path.coords[0])
        )
        if circle_cutout:
            continue_around = gt.ring_with_origin_at(
                headland_shape,
                Point(curve2.path.coords[0])
            )
            circled_coords = (
                list(headland_segment.coords) +
                list(continue_around.coords)
            )
            headland_segment = LineString(circled_coords)

        combined_coords = (
            list(headland_segment.coords[0:-1])
            + list(curve2.path.coords)
        )
    elif not curve2.valid:
        # part 2 not valid
        # save part 1, circle cutouts if needed and combine coords
        valid = False
        headland_segment = gt.get_substring_on_linearring(
            headland_shape,
            Point(curve1.path.coords[-1]),
            to_node.intersect_secondary.position
        )
        if circle_cutout:
            continue_around = gt.ring_with_origin_at(
                headland_shape,
                Point(curve1.path.coords[-1])
            )
            circled_coords = (
                list(headland_segment.coords) +
                list(continue_around.coords)
            )
            headland_segment = LineString(circled_coords)

        combined_coords = (
            list(curve1.path.coords)
            + list(headland_segment.coords[1:])
        )
    else:
        # curve completely invalid
        # return start point --> end point as last resort path
        if route_params.debug_prints and not valid:
            print("connect_ab_lines() was completely invalid")
        path = [from_node.position, to_node.position]
        return Curve(path, CurveType.U_TURN, False)

    points = [Point(coord) for coord in combined_coords]

    if route_params.debug_prints and not valid:
        print("connect_ab_lines() was not valid")

    curve_type = Curve.get_dominant_curve_type(curve1, curve2)
    return Curve(points, curve_type, valid)


def trace_headland_hop(
        nodes,
        route_params: RoutePlanningConfig,
        from_index,
        to_index,
        headlands,
        field_border,
        inner_field_border,
        circled_cutouts: Dict[int, bool] | None = None) -> Curve:
    """
    Traces a path from a cutout headland to the outer headland if there are only ab lines on one side of the cutout.
    """

    from_node = nodes[from_index]
    if to_index == -1:
        to_index = len(nodes) - 1
    if from_index >= to_index:
        raise KeyError("trace_headland_hop() called with inappropriate indexes")

    # disable working corridor filling for now
    last_fill_corridors = route_params.corridor_strategy
    route_params.corridor_strategy = CorridorStrategy.DRIVE_NONE

    # construct imaginary ab line to traverse to the other headland ring
    from_ring = abs(from_node.ring_index)
    for i in range(from_index + 1, to_index + 1):
        node = nodes[i]
        ring_index = (node.ring_index)
        if ring_index != from_ring:
            n1 = PrimaryTrackGraphNode(nodes[i - 1].position, from_ring)
            n2 = PrimaryTrackGraphNode(node.position, ring_index)
            n1.link_primary_neighbor(n2, route_params)

            sorted_headlands = headlands.copy()
            sorted_headlands.sort(key=lambda ring: ring.distance(n1.position))
            from_headland = sorted_headlands[0]

            # construct ab parallel line for actual traversal
            ab_vector = gt.direction_of_line(from_node.get_ab_line())
            if from_headland == headlands[0]:
                projected_point = Point(
                    (n2.position.x + ab_vector[0], n2.position.y + ab_vector[1])
                )
                ab_vector = LineString([n2.position, projected_point])
                node_to_align = n1
            else:
                projected_point = Point(
                    (n1.position.x + ab_vector[0], n1.position.y + ab_vector[1])
                )
                ab_vector = LineString([n1.position, projected_point])
                node_to_align = n2

            ab_vector = gt.extend_line_in_bounds(ab_vector, headlands[0])
            ab_vector = gt.extend_line(
                LineString([ab_vector.coords[0], ab_vector.coords[-1]]),
                0.1
            )

            vector_intersect = ab_vector.intersection(headlands[0])
            if isinstance(vector_intersect, Point):
                node_to_align.position = vector_intersect
            elif isinstance(vector_intersect, MultiPoint):
                points = list(vector_intersect.geoms)
                points.sort(key=lambda point: point.distance(n1.position))
                node_to_align.position = points[0]

            line = LineString([n1.position, n2.position])
            n1.set_ab_line(line)
            n2.set_ab_line(line)
            node_tree_1 = STRtree([nodes[i - 1].position])
            node_tree_2 = STRtree([node.position])
            n1.set_secondaries([nodes[i - 1]], route_params, node_tree_1, Polygon(field_border))
            n2.set_secondaries([node], route_params, node_tree_2, Polygon(field_border))

            nodes_cpy = nodes[from_index:i + 1] + [n1, n2] + nodes[i:to_index + 1]
            new_from_index = 0
            new_to_index = i + 1 - from_index

            part1 = connect_ab_lines(
                nodes_cpy,
                route_params,
                new_from_index,
                new_to_index,
                headlands,
                field_border,
                inner_field_border,
                False,
                circled_cutouts
            )
            part1_line = LineString(part1.path)

            if part1_line.length > headlands[0].length * 0.9:
                part1 = trace_neighbor_curve(
                    nodes_cpy,
                    new_from_index,
                    new_to_index,
                    headlands,
                    field_border,
                    inner_field_border,
                    route_params
                )
                part1_line = LineString(part1.path)

            new_from_index = new_to_index + 1
            new_to_index = -1

            part2 = connect_ab_lines(
                nodes_cpy,
                route_params,
                new_from_index,
                new_to_index,
                headlands,
                field_border,
                inner_field_border,
                False,
                circled_cutouts
            )
            part2_line = part2.path

            if part2_line.length > headlands[0].length * 0.9:
                part2 = trace_neighbor_curve(
                    nodes_cpy,
                    new_from_index,
                    new_to_index,
                    headlands,
                    field_border,
                    inner_field_border,
                    route_params
                )
                part2_line = LineString(part2.path)

            route_params.corridor_strategy = last_fill_corridors
            return Curve(
                list(part1.path.coords) + list(part2.path.coords),
                CurveType.U_TURN,
                True
            )


def trace_neighbor_curve(
        nodes,
        from_index,
        to_index,
        headlands,
        field_border,
        inner_field_border,
        route_params: RoutePlanningConfig) -> Curve:
    """
    Returns Curve to an ab line directly neighboring the other.

    Uses the relations of track graph nodes to trace a curve between two neighboring primary nodes over a headland shape using
    the pi-curve.
    Returns the dubins path from the starting node to the target node and wether or not the path is valid.
    """
    from_node: PrimaryTrackGraphNode = nodes[from_index]
    to_node: PrimaryTrackGraphNode = nodes[to_index]

    if route_params.disable_pi_curves:
        print("\n\nReturning disabled pi curve\n")
        return Curve(
            [
                from_node.position,
                to_node.position
            ],
            CurveType.UNDEFINED,
            False
        )

    sorted_headlands = headlands.copy()
    sorted_headlands.sort(
        key=lambda ring: ring.distance(
            from_node.intersect_secondary.position
        )
    )
    headland = sorted_headlands[0]

    # check headland run
    crossing_point_1 = from_node.intersect_secondary.position
    crossing_point_2 = to_node.intersect_secondary.position
    headland_with_origin = gt.ring_with_origin_at(headland, crossing_point_1)
    if _is_long_way_around(headland_with_origin, crossing_point_2):
        headland = headland.reverse()

    headland = gt.ring_with_origin_at(
        headland,
        headland.interpolate(
            headland.project(crossing_point_1) - route_params.direction_change_extension_distance * 5
        )
    )

    from_segment = from_node.get_ab_line()
    headland_segment = gt.get_substring_on_linearring(
        headland,
        headland.interpolate(
            headland.project(crossing_point_1) - route_params.direction_change_extension_distance * 5
        ),
        headland.interpolate(
            headland.project(crossing_point_2) + route_params.direction_change_extension_distance * 5
        )
    )

    is_at_cutout = from_node.ring_index != 0

    curve1 = search_curve_to_headland(
        from_segment,
        headland_segment,
        headland,
        field_border,
        inner_field_border,
        is_at_cutout,
        route_params
    )

    to_segment = to_node.get_ab_line().reverse()
    curve2 = search_curve_to_ab(
        to_segment,
        headland_segment,
        headland,
        field_border,
        inner_field_border,
        is_at_cutout,
        route_params
    )

    if not curve1.valid or not curve2.valid:
        return Curve([from_node.position, to_node.position], CurveType.PI_CURVE, False)

    headland_connection_middle = gt.get_substring_on_linearring(
        headland,
        Point(curve2.path.coords[0]),
        Point(curve1.path.coords[-1])
    )
    headland_connection_middle = headland_connection_middle.reverse()

    headland_connection_fwd = gt.get_substring_on_linearring(
        headland,
        Point(headland_connection_middle.coords[0]),
        headland.interpolate(
            headland.project(
                Point(headland_connection_middle.coords[0])
            ) + route_params.direction_change_extension_distance
        )
    )

    headland_connection_bkwd = gt.get_substring_on_linearring(
        headland,
        headland.interpolate(
            headland.project(
                Point(headland_connection_middle.coords[-1])
            ) - route_params.direction_change_extension_distance
        ),
        Point(headland_connection_middle.coords[-1])
    )

    headland_connection_coords = (
        list(headland_connection_fwd.coords) +
        list(headland_connection_fwd.reverse().coords) +
        list(headland_connection_middle.coords) +
        list(headland_connection_bkwd.reverse().coords) +
        list(headland_connection_bkwd.coords)
    )

    combined_coords = (
        list(curve1.path.coords)
        + headland_connection_coords
        + list(curve2.path.coords)
    )
    path = [Point(coord) for coord in combined_coords]
    return Curve(path, CurveType.PI_CURVE, True)


def search_curve_to_headland(
        ab_line: LineString,
        headland_segment: LineString,
        headland_ring: LineString,
        field_border: LinearRing,
        inner_field_border: LinearRing,
        is_at_cutout: bool,
        route_params: RoutePlanningConfig,
        metrics: EdgeMetrics = None) -> Curve:
    """
    Searches for the most favorable turn from the end of an AB line onto a headland shape.

    Returns the curve as LineString.
    """
    working_corridor = get_corridor_line(
        ab_line,
        route_params.working_width,
        headland_ring,
        inner_field_border,
        route_params.implement_working_offset
    )

    # try if simple turn fits
    path = get_simple_turn_to_headland(
        ab_line,
        headland_segment,
        headland_ring,
        field_border,
        route_params
    )

    if path is None:
        if route_params.debug_prints:
            print("Did not find path to headland")

        p1, p2 = nearest_points(working_corridor, headland_segment)
        path = [p1, p2]
        return Curve(path, CurveType.U_TURN, False)

    curve_type = CurveType.U_TURN
    valid = True

    curve_end_projection = working_corridor.project(Point(path.coords[0]))
    curve_end_projection = working_corridor.length - curve_end_projection

    # is the end of the path within a given range of the start of the working corridor?
    corridor_error_detected: bool = not metrics or metrics.working_corridor_error > 0
    if curve_end_projection > route_params.corridor_threshold and corridor_error_detected:
        # should this working corridor be driven?
        strategy = route_params.corridor_strategy
        if (strategy == CorridorStrategy.DRIVE_ALL or
                strategy == CorridorStrategy.DRIVE_ONLY_OUTER_HEADLAND and not is_at_cutout):
            if route_params.corridor_curve == CurveType.HOOK:
                path = insert_hook_stops_to_headland(
                    path,
                    working_corridor,
                    headland_ring,
                    route_params
                )
                curve_type = CurveType.HOOK
                valid = True
            elif route_params.corridor_curve == CurveType.LOOP:
                return search_loop_curve(
                    working_corridor,
                    headland_segment,
                    headland_ring,
                    field_border,
                    is_at_cutout,
                    True,
                    route_params
                )

    if not path.dwithin(headland_ring, 0.1):
        path = gt.extend_line_in_bounds(path, headland_ring, extend_back=False)

    return Curve(path, curve_type, valid)


def search_curve_to_ab(
        ab_line: LineString,
        headland_segment: LineString,
        headland_ring: LineString,
        field_border: LinearRing,
        inner_field_border: LinearRing,
        is_at_cutout: bool,
        route_params: RoutePlanningConfig,
        metrics: EdgeMetrics = None) -> Curve:
    """
    Searches for the most favorable turn from a headland shape onto an AB line.

    Returns the curve as LineString.
    """

    working_corridor = get_corridor_line(
        ab_line,
        route_params.working_width,
        headland_ring,
        inner_field_border,
        route_params.implement_working_offset
    )

    # try if simple turn fits
    path = get_simple_turn_to_ab(
        ab_line,
        headland_segment,
        headland_ring,
        field_border,
        route_params
    )

    if path is None:
        if route_params.debug_prints:
            print("Did not find curve from headland to ab")

        p1, p2 = nearest_points(headland_segment, working_corridor)
        path = [p1, p2]
        return Curve(path, CurveType.U_TURN, False)

    curve_type = CurveType.U_TURN
    valid = True

    curve_end_projection = working_corridor.project(Point(path.coords[-1]))

    # is the end of the curve within a given range of the start of the working corridor?
    corridor_error_detected: bool = not metrics or metrics.working_corridor_error > 0
    if curve_end_projection > route_params.corridor_threshold and corridor_error_detected:
        # should this working corridor be driven?
        strategy = route_params.corridor_strategy
        if (strategy == CorridorStrategy.DRIVE_ALL or
                strategy == CorridorStrategy.DRIVE_ONLY_OUTER_HEADLAND and not is_at_cutout):

            if route_params.corridor_curve == CurveType.HOOK:
                path = insert_hook_stops_to_ab(
                    path,
                    working_corridor,
                    headland_ring,
                    route_params
                )
                curve_type = CurveType.HOOK
                valid = True
            elif route_params.corridor_curve == CurveType.LOOP:
                return search_loop_curve(
                    working_corridor,
                    headland_segment,
                    headland_ring,
                    field_border,
                    is_at_cutout,
                    False,
                    route_params
                )

    if not path.dwithin(headland_ring, 0.1):
        path = gt.extend_line_in_bounds(path, headland_ring, extend_front = False)

    return Curve(path, curve_type, valid)


def get_simple_turn_to_ab(
        ab_line: LineString,
        headland_segment: LineString,
        headland_ring: LineString,
        field_border: LinearRing,
        route_params: RoutePlanningConfig) -> LineString | None:
    """
    Returns the path connecting the end of an ab line to the given headland segment if a simple turn fits between the segments.
    """

    curve: LineString | None = None
    if not route_params.robust_curve_calculation_only:
        # Try simple solution
        curve = gt.turn_to_segment(
            headland_segment,
            ab_line,
            field_border,
            route_params.vehicle_turning_radius,
            False
        )
        if curve is not None:
            return curve

    # try robust solution
    curve_start = None
    extended_headland_segment = gt.extend_line(headland_segment, route_params._track_width)
    curve_start_candidates = [Point(coord) for coord in extended_headland_segment.coords[0:-1][::-1]]

    candidate_index = 0
    while ((curve is None or curve.length > route_params.vehicle_turning_radius * math.pi)
            and candidate_index <= len(curve_start_candidates) - 1):
        curve_start = headland_ring.interpolate(
            headland_ring.project(curve_start_candidates[candidate_index])
        )

        curve = gt.dubins_between_segments(
            extended_headland_segment,
            ab_line,
            route_params.vehicle_turning_radius,
            current_point=curve_start,
            bounds=field_border
        )

        candidate_index += 1
        if candidate_index > len(curve_start_candidates) - 1:
            break
        next_candidate = curve_start_candidates[candidate_index]
        while (curve_start.distance(next_candidate) < 0.5
               and candidate_index < len(curve_start_candidates) - 1):
            candidate_index += 1
            next_candidate = curve_start_candidates[candidate_index]

    if curve is None or curve.length > route_params.vehicle_turning_radius * math.pi:
        return None

    return curve


def get_simple_turn_to_headland(
        ab_line: LineString,
        headland_segment: LineString,
        headland_ring: LineString,
        field_border: LinearRing,
        route_params: RoutePlanningConfig) -> LineString | None:
    """
    Calculates a turn to headland from an ab line.
    """

    target_tangent = gt.get_tangent_at_nearest_point(ab_line.interpolate(1.0, True), headland_ring)
    # TODO calculate with turning radius
    if abs(gt.angle_between_lines(ab_line, target_tangent)) < pi / 10.0 and target_tangent.dwithin(ab_line, 0.5):
        start_point = ab_line.interpolate(
            ab_line.length - 5.0
        )
        return LineString(
            [
                start_point,
                headland_ring.interpolate(
                    headland_ring.project(Point(ab_line.coords[-1]))
                )
            ]
        )

    curve: LineString | None = None
    if not route_params.robust_curve_calculation_only:
        # Try simple solution
        curve = gt.turn_to_segment(
            ab_line,
            headland_segment,
            field_border,
            route_params.vehicle_turning_radius,
            True
        )
        if curve is not None:
            return curve

    # Try robust solution
    curve_start = ab_line.interpolate(1.0, True)
    skip = 0

    while (curve is None and skip < 3 or (curve is not None and
           curve.length > route_params.vehicle_turning_radius * math.pi and skip < 3)):
        if len(headland_segment.coords[skip:]) < 2:
            break

        headland_segment = LineString(headland_segment.coords[skip:])

        curve_start_projection = 1.0
        continue_searching = True
        while continue_searching:
            curve_start = ab_line.interpolate(curve_start_projection, True)
            curve = gt.dubins_between_segments(
                ab_line,
                headland_segment,
                route_params.vehicle_turning_radius,
                current_point=curve_start,
                bounds=field_border
            )
            curve_start_projection -= 0.1

            continue_searching = (
                (curve is None and curve_start_projection > 0.1) or
                (curve is not None and curve.length > route_params.vehicle_turning_radius * math.pi
                    and curve_start_projection > 0.0))

        skip += 1

    if curve is not None and curve.length > route_params.vehicle_turning_radius * math.pi:
        return None

    return curve


def _is_long_way_around(ring: LinearRing, point: Point) -> bool:
    return ring.project(point, True) > 0.5


def trace_curve(
        nodes: List[PrimaryTrackGraphNode | SecondaryTrackGraphNode],
        from_index: int,
        to_index: int,
        headlands: List[LinearRing],
        field_border,
        inner_field_border,
        route_params) -> Curve:
    """
    Uses the relations of track graph nodes to trace a curve between nodes over a headland shape.

    one or both of start or end node have to be on headland.
    Returns the dubins path from the starting node to the target node.
    """

    if to_index >= len(nodes):
        raise KeyError("trace_curve() was called with inappropriate indexes")
    if to_index == -1:
        to_index = len(nodes) - 1
    if from_index == to_index:
        raise KeyError("trace_curve() was called with inappropriate indexes")

    from_node = nodes[from_index]
    to_node = nodes[to_index]
    if isinstance(from_node, PrimaryTrackGraphNode) and isinstance(to_node, PrimaryTrackGraphNode):
        raise ValueError(
            "Tried to trace curve with u turn params. "
            + "If smoothing a path from a primary node to another primary node "
            + "connect_ab_lines should be called."
        )

    sorted_headlands = headlands.copy()
    sorted_headlands.sort(key=lambda headland: from_node.position.distance(headland))
    headland_shape = sorted_headlands[0]

    # gather locations and generate directed segments to traverse
    ab_segment = None
    if isinstance(from_node, PrimaryTrackGraphNode):
        ab_segment = from_node.get_ab_line()
    elif isinstance(to_node, PrimaryTrackGraphNode):
        ab_segment = to_node.get_ab_line().reverse()

    # look for nodes to construct headland tangent
    over_node = None
    for i in range(from_index, len(nodes)):
        if isinstance(nodes[i], SecondaryTrackGraphNode):
            over_node = nodes[i]
            segment_tangent_start = over_node.position
            if i < len(nodes) - 1:
                if isinstance(nodes[i + 1], PrimaryTrackGraphNode):
                    over_extend_node = nodes[i + 1].intersect_secondary
                else:
                    over_extend_node = nodes[i + 1]
                segment_tangent_end = over_extend_node.position
            else:
                raise ValueError("curve was to short and could not find headland tangent")
            break

    # check direction of headland against the segment tangent
    headland_with_origin = gt.ring_with_origin_at(
        headland_shape,
        headland_shape.interpolate(headland_shape.project(segment_tangent_start))
    )
    if _is_long_way_around(headland_with_origin, segment_tangent_end):
        # if run of headland shape is other way reverse it
        headland_shape = headland_shape.reverse()

    if isinstance(from_node, SecondaryTrackGraphNode) and isinstance(to_node, SecondaryTrackGraphNode):
        substr = gt.get_substring_on_linearring(
            headland_shape,
            from_node.position,
            to_node.position
        )
        points = [Point(c) for c in substr.coords]
        return Curve(points, CurveType.UNDEFINED, True)
    elif isinstance(from_node, SecondaryTrackGraphNode):
        cut_start = headland_shape.interpolate(
            headland_shape.project(
                from_node.position
            ) - 10.0
        )
        cut_end = headland_shape.interpolate(
            headland_shape.project(
                to_node.intersect_secondary.position
            ) + 10.0
        )
        headland_segment = gt.get_substring_on_linearring(
            headland_shape,
            cut_start,
            cut_end
        )
        if isinstance(headland_segment, Point):
            # if this happens just do a roundtrip
            headland_segment = gt.ring_with_origin_at(headland_shape, from_node.intersect_secondary.position)
            headland_segment = gt.substring(
                headland_segment,
                0,
                headland_segment.length
            )

        is_at_cutout = from_node != 0 and to_index != 0
        curve = search_curve_to_ab(
            ab_segment,
            headland_segment,
            headland_shape,
            field_border,
            inner_field_border,
            is_at_cutout,
            route_params,
            from_node.edges[to_node.index]
        )
        if curve.valid:
            headland_part = gt.get_substring_on_linearring(
                headland_shape,
                from_node.position,
                Point(curve.path.coords[0])
            )
            coords = list(headland_part.coords) + list(curve.path.coords)
        else:
            return Curve(
                [from_node.position, to_node.position],
                CurveType.UNDEFINED,
                False
            )

    elif isinstance(to_node, SecondaryTrackGraphNode):
        cut_start = headland_shape.interpolate(
            headland_shape.project(
                from_node.intersect_secondary.position
            ) - 10.0
        )
        cut_end = headland_shape.interpolate(
            headland_shape.project(
                to_node.position
            ) + 10.0
        )
        headland_segment = gt.get_substring_on_linearring(
            headland_shape,
            cut_start,
            cut_end
        )
        is_at_cutout = from_node != 0 and to_index != 0
        curve = search_curve_to_headland(
            ab_segment,
            headland_segment,
            headland_shape,
            field_border,
            inner_field_border,
            is_at_cutout,
            route_params,
            from_node.edges[to_node.index]
        )
        if curve.valid:
            curve_end_projection = headland_shape.project(
                Point(curve.path.coords[-1])
            )
            target_projection = headland_shape.project(
                to_node.position
            )
            if curve_end_projection < target_projection:
                headland_part = gt.get_substring_on_linearring(
                    headland_shape,
                    Point(curve.path.coords[-1]),
                    to_node.position
                )
                coords = list(curve.path.coords) + list(headland_part.coords)
            else:
                coords = list(curve.path.coords)
        else:
            return Curve(
                [from_node.position, to_node.position],
                CurveType.UNDEFINED,
                False
            )

    points = [Point(c) for c in coords]
    return Curve(points, CurveType.UNDEFINED, True)


def insert_hook_stops_to_ab(
        curve: LineString,
        working_corridor,
        turning_headland,
        route_params: RoutePlanningConfig):
    """
    Inserts the needed points in a curve from a headland segment onto an ab line working corridor for a hook curve.
    """

    points = [Point(c) for c in curve.coords]
    curve_end = points[-1]

    curve_end_projection = working_corridor.reverse().project(
        curve_end,
    )
    if working_corridor.length - curve_end_projection > route_params.corridor_threshold:
        extension_point = working_corridor.interpolate(
            working_corridor.project(curve_end) + route_params.direction_change_extension_distance
        )
        points.append(extension_point)

        if route_params.working_corridor_extension:
            if working_corridor is None or not isinstance(working_corridor, LineString) or working_corridor.length < 0.01:
                if route_params.debug_prints:
                    print("Working corridor is very short, extension failed")
                return curve
            else:
                working_corridor = gt.extend_line_in_bounds(
                    working_corridor,
                    Polygon(turning_headland),
                    route_params.direction_change_extension_distance,
                    extend_front=False,
                    extend_back=True,
                )
        working_corridor_oriented = working_corridor.intersection(Polygon(turning_headland))
        if isinstance(working_corridor_oriented, MultiLineString):
            for line in working_corridor_oriented.geoms:
                if line.length > 0.1:
                    working_corridor_oriented = line
                    break

        insert_point = Point(working_corridor_oriented.coords[0])
        points.append(insert_point)
        return LineString(points)

    return curve


def insert_hook_stops_to_headland(
        curve: LineString,
        working_corridor,
        turning_headland,
        route_params: RoutePlanningConfig):
    """
    Inserts the needed points in a curve from an ab line working corridor onto the headland for a hook curve.
    """
    points = [Point(c) for c in curve.coords]
    curve_start = points[0]

    curve_start_projection = working_corridor.project(
        curve_start,
    )

    if working_corridor.length - curve_start_projection > route_params.corridor_threshold:
        backup_point = working_corridor.interpolate(
            working_corridor.project(
                curve_start
            ) - route_params.direction_change_extension_distance
        )
        points.insert(0, backup_point)
        if working_corridor is None or not isinstance(working_corridor, LineString) or working_corridor.length < 0.01:
            if route_params.debug_prints:
                print("Working corridor is very short, extension failed")
            return curve
        else:
            working_corridor = gt.extend_line_in_bounds(
                working_corridor,
                Polygon(turning_headland),
                route_params.direction_change_extension_distance,
                extend_front=True,
                extend_back=False,
            )
        working_corridor = working_corridor.intersection(Polygon(turning_headland))
        if isinstance(working_corridor, MultiLineString):
            for line in working_corridor.geoms:
                if line.length > 0.1:
                    working_corridor = line
                    break
        corridor_end = Point(working_corridor.coords[-1])
        points.insert(0, corridor_end)
        return LineString(points)

    return curve


def search_loop_curve(
       working_corridor: LineString,
       headland_segment: LineString,
       headland_ring: LinearRing,
       field_border: LinearRing,
       is_at_cutout: bool,
       from_ab_to_head,
       route_params: RoutePlanningConfig) -> Curve:
    """
    Returns a looping curve from a working corridor line to a headland segment in the given direction.
    """

    headland_ring_aligned = LinearRing(headland_ring)
    working_corridor_aligned = LineString(working_corridor)
    headland_segment_aligned = LineString(headland_segment)
    if not from_ab_to_head:
        headland_ring_aligned = headland_ring.reverse()
        working_corridor_aligned = working_corridor.reverse()
        headland_segment_aligned = headland_segment.reverse()

    # turn to the other direction onto the headland for curve part 1
    headland_segment_end_projection = headland_ring_aligned.project(
        Point(headland_segment_aligned.coords[-1])
    )
    headland_segment_start_projection = headland_ring_aligned.project(
        Point(headland_segment_aligned.coords[0])
    )
    part1_headland_segment = gt.get_substring_on_linearring(
        headland_ring_aligned,
        headland_ring_aligned.interpolate(headland_segment_start_projection - 50.0),
        headland_ring_aligned.interpolate(headland_segment_end_projection)
    ).reverse()

    part1_ab_cut = LineString(
        [
            Point(working_corridor_aligned.coords[0]),
            working_corridor_aligned.interpolate(0.75, True)
        ]
    )
    part1 = get_simple_turn_to_headland(
        part1_ab_cut,
        part1_headland_segment,
        headland_ring_aligned,
        field_border,
        route_params
    )

    if part1 is None:
        p1, p2 = nearest_points(working_corridor, headland_segment_aligned)
        path = [
            p1,
            p2
        ]

        if not from_ab_to_head:
            path.reverse()
        if route_params.debug_prints:
            print("search_loop_curve() failed")

        return Curve(path, CurveType.HOOK, False)

    # extend on the headland for a few meters
    reverse_headland_ring = headland_ring_aligned.reverse()
    extend_path_start = Point(part1.coords[-1])

    connection_found = False
    extension = 0
    # try how much distance has to be extended on headland before connection fits in place
    while not connection_found:
        headland_extend_path = gt.get_substring_on_linearring(
            reverse_headland_ring,
            extend_path_start,
            reverse_headland_ring.interpolate(
                reverse_headland_ring.project(
                    extend_path_start
                )
                + route_params.loop_curve_initial_extend
                + (extension + 0.1 * pow(extension, 2.0))
            )
        )

        # get a target vector to plan the loop towards
        tangent_point = Point(headland_extend_path.coords[-1])
        headland_tangent = gt.get_tangent_at_nearest_point(tangent_point, headland_ring_aligned)
        tangent_r = headland_tangent.parallel_offset(
            2 * route_params._track_width,
            "right"
        )
        tangent_l = headland_tangent.parallel_offset(
            2 * route_params._track_width,
            "left"
        )
        part2_target = tangent_r
        headland_test_poly = Polygon(headland_ring_aligned)
        headland_contains_r = headland_test_poly.contains(tangent_r)
        if is_at_cutout:
            if headland_contains_r:
                part2_target = tangent_l
        else:
            if not headland_contains_r:
                part2_target = tangent_l

        part2 = gt.dubins_between_vectors(
            headland_tangent.reverse(),
            part2_target,
            route_params._track_width
        )
        part3_in_direction = LineString([part2.coords[-2], part2.coords[-1]])
        look_ahead_point_on_line = headland_ring_aligned.interpolate(
            headland_ring_aligned.project(
                tangent_point
            ) + route_params.loop_curve_lookahead
        )
        part3 = gt.dubins.shortest_path(
            (  # in vector
                part3_in_direction.coords.xy[0][1],
                part3_in_direction.coords.xy[1][1],
                gt.angle_of_line(part3_in_direction),
            ),
            (  # target vector
                look_ahead_point_on_line.x,
                look_ahead_point_on_line.y,
                gt.angle_of_line(
                    gt.get_tangent_at_nearest_point(
                        look_ahead_point_on_line,
                        headland_ring_aligned
                    )
                )
            ),
            route_params.vehicle_turning_radius
        ).sample_many(0.5)[0]
        part3_points = []
        for coord in part3:
            part3_points.append(Point(coord[0], coord[1]))
        part3_points.append(look_ahead_point_on_line)

        connection = LineString(part3_points)
        if connection.length < 55:
            break
        elif extension > 40:
            # return None?
            if route_params.debug_prints:
                print("didn't find fitting loop curve")
            break

        extension += 1

    coords = LineString(
        list(part1.coords) +
        list(headland_extend_path.coords) +
        list(part2.coords) +
        part3_points
    )

    if not from_ab_to_head:
        coords = coords.reverse()

    return Curve(coords, CurveType.LOOP, True)
