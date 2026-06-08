from math import pi
from typing import List, Tuple

import numpy as np
from shapely import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    oriented_envelope,
    remove_repeated_points,
    unary_union,
)
from shapely.ops import linemerge, nearest_points

from nexat_trace.planning.track_graph.secondary_track_graph_node import SecondaryTrackGraphNode
from nexat_trace.shared.config import RoutePlanningConfig
from nexat_trace.shared.exceptions import GraphConstructionError
from nexat_trace.shared.planning_messages import PlanningMsg
from nexat_trace.track_system import TrackSystem
from nexat_trace.util import geom_tools as gt


def get_target_headland_from_track_system(
        track_system: TrackSystem,
        config: RoutePlanningConfig) -> Tuple[LinearRing | List[LinearRing], bool]:
    """
    Returns the target headlands depending on the current working width and wether or not it is the configured index.

    Target headlands are rounded to vehicle turning radius depending on the given working width.
    """

    target_headland = 1
    if config.override_headland_index is not None:
        target_headland = config.override_headland_index
    else:
        if config.working_width >= 4 * config._track_width:
            target_headland = 2

        if config.working_width >= 5 * config._track_width:
            target_headland = 3

    is_target = target_headland < len(track_system.headland_config)

    ring_index = min(len(track_system.headlands) - 1, target_headland)
    headland_prep = _get_headland_prep(track_system, config)

    border_poly = track_system.outer_border
    rings = []
    _offset = round(track_system.outer_border.exterior.distance(track_system.headlands[0][0]) - headland_prep[0], 1)
    headland_prep_sum = sum(headland_prep[:ring_index + 1]) + _offset
    outer = Polygon(border_poly.exterior).buffer(-1 * headland_prep_sum, resolution=45, cap_style=2, join_style=2)
    rings.append(gt.erode_linearring(outer.exterior, config.vehicle_turning_radius))
    for inner in border_poly.interiors:
        if Polygon(inner).area > 200.0:
            inner = inner.buffer(headland_prep_sum, resolution=45, cap_style=2, join_style=2)
            inner = gt.erode_linearring(inner.exterior, config.vehicle_turning_radius)
            rings.append(inner)

    outer = Polygon(rings[0])
    holes = rings[1:]

    # Iteratively subtract intersecting holes
    changed = True
    while changed:
        changed = False
        remaining = []
        for hole in holes:
            if outer.exterior.intersects(hole):
                outer = outer.difference(Polygon(hole))
                changed = True  # shape changed, need to re-check remaining holes
            else:
                remaining.append(hole)
        holes = remaining

    poly = gt.erode_polygon_inwards(outer, config.vehicle_turning_radius)
    poly = poly.simplify(1e-3, preserve_topology=True)
    rounded_rings = [poly.exterior] + list(poly.interiors) + holes
    return rounded_rings, is_target


def get_secondary_positions(
        ab_lines: List[LineString],
        headlands: List[LinearRing],
        route_params: RoutePlanningConfig) -> List[SecondaryTrackGraphNode]:
    """
    Returns the positions where secondary nodes should be placed on the headland rings.
    """
    headland_collection = GeometryCollection(headlands)
    cut_ab_lines = []
    for line in ab_lines:
        if line.intersects(headland_collection) and line.length > 2.1:
            cut_line = gt.substring(line, 1.0, line.length - 1.0)
            cut_ab_lines.append(cut_line)
        else:
            cut_ab_lines.append(line)

    points = []
    multi_ab_lines = MultiLineString(
        [gt.extend_line(line, 100000.0) for line in ab_lines]
    )
    colinear_extension_points = []

    multi_ab_lines_unextended = MultiLineString(cut_ab_lines)

    for line in cut_ab_lines:
        # Check front of line for collisions
        forward_line = LineString(line)
        forward_end = Point(line.coords[-1])

        intersections: List[SecondaryTrackGraphNode] = []
        for i in range(0, len(headlands)):
            headland = headlands[i]
            extended_line = gt.extend_line_in_bounds(forward_line, headlands[0], extend_back = False)
            # extend line beyond headland ring to make clear intersection
            extended_line = gt.extend_line(extended_line, 1.0, extend_back = False)

            intersection = extended_line.intersection(headland)
            if isinstance(intersection, MultiPoint):
                intersections.extend([SecondaryTrackGraphNode(point, i) for point in intersection.geoms])
            elif isinstance(intersection, Point):
                intersections.append(SecondaryTrackGraphNode(intersection, i))

        if len(intersections) >= 1:
            intersections.sort(key=lambda p: p.distance_to(forward_end))
            points.append(intersections[0])

            # 3 points, all within a certain distance_to of each other (min/max to headland)
            # 4 * track_width * 1.5 max probable working width
            if (len(intersections) == 3 and
                intersections[1].distance_to(intersections[2]) > route_params.vehicle_turning_radius and
                intersections[1].distance_to(intersections[2]) < 4 * route_params._track_width * 1.5):

                # is there an ab line here?
                link_edge_candidate = LineString([intersections[1].position, intersections[2].position])
                link_edge_candidate = LineString(
                    [
                        link_edge_candidate.interpolate(0.1, True),
                        link_edge_candidate.interpolate(0.9, True)
                    ]
                )
                if not link_edge_candidate.dwithin(multi_ab_lines_unextended, 0.01):
                    intersections[1].link_node(intersections[2], None, None)
                    points.extend(intersections[1:])

        else:
            print(f"intersections for secondary positions malformed: {intersections}")
            raise GraphConstructionError(
                "Could not determine positions for secondary nodes. "
                "This may be because of the working width in combination with "
                "cutouts on the outer headland"
            )

        # Check back of line for collisions
        backward_line = line.reverse()
        back_end = Point(line.coords[0])

        intersections = []
        for i in range(0, len(headlands)):
            headland = headlands[i]
            extended_line = gt.extend_line_in_bounds(backward_line, headlands[0], extend_back = False)
            # extend line beyond headland ring to make clear intersection
            extended_line = gt.extend_line(extended_line, 1.0, extend_back = False)

            intersection = extended_line.intersection(headland)
            if isinstance(intersection, MultiPoint):
                intersections.extend([SecondaryTrackGraphNode(point, i) for point in intersection.geoms])
            elif isinstance(intersection, Point):
                intersections.append(SecondaryTrackGraphNode(intersection, i))

        if len(intersections) >= 1:
            intersections.sort(key=lambda p: p.distance_to(back_end))
            points.append(intersections[0])

            # 3 points, all within a certain distance_to of each other (min/max to headland)
            # 4 * track_width * 1.5 max probable working width
            if (len(intersections) == 3 and
                intersections[1].distance_to(intersections[2]) > route_params.vehicle_turning_radius and
                intersections[1].distance_to(intersections[2]) < 4 * route_params._track_width * 1.5):
                # is there an ab line here?
                link_edge_candidate = LineString([intersections[1].position, intersections[2].position])
                link_edge_candidate = LineString(
                    [
                        link_edge_candidate.interpolate(0.1, True),
                        link_edge_candidate.interpolate(0.9, True)
                    ]
                )
                if not link_edge_candidate.dwithin(multi_ab_lines_unextended, 0.01):
                    intersections[1].link_node(intersections[2], None, None)
                    points.extend(intersections[1:])

        else:
            print(f"intersections for secondary positions malformed: {intersections}")
            raise GraphConstructionError(
                "Could not determine positions for secondary nodes. "
                "This may be because of the working width in combination with "
                "cutouts on the outer headland"
            )

        # Create colinear extension points
        for i in range(0, len(headlands)):
            if not extended_line.intersects(headlands[i]):
                continue

            over_extended_line = gt.extend_line(extended_line, 100000.0)
            intersection = over_extended_line.intersection(headlands[i])
            if isinstance(intersection, Point):
                p = intersection
            elif isinstance(intersection, MultiPoint):
                p = intersection.geoms[0]
            else:
                continue

            headland = gt.ring_with_origin_at(
                headlands[i],
                p
            )
            difference = headland.difference(multi_ab_lines)
            if not isinstance(difference, MultiLineString):
                continue
            slices = list(difference.geoms)

            for slice_line in slices:
                if slice_line.length < route_params._track_width:
                    continue
                p1 = Point(slice_line.coords[0])
                p2 = Point(slice_line.coords[-1])
                if p1.dwithin(over_extended_line, 1.0) and p2.dwithin(over_extended_line, 1.0):
                    middle = slice_line.interpolate(0.5, True)
                    if all(middle.distance(other.position) > 1.0 for other in colinear_extension_points):
                        colinear_extension_points.append(SecondaryTrackGraphNode(middle, i))

    points.extend(colinear_extension_points)

    # Sort secondary points into rings
    rings = []
    for i, headland in enumerate(headlands):
        points_on_ring = []
        for point in points:
            if point.ring_index == i:
                points_on_ring.append(point)

        # sort the positions after the run of the original headland ring
        points_on_ring.sort(key=lambda point: headland.project(point.position))

        rings.append(points_on_ring)

    connect_rings(headlands, rings, route_params)

    # return flattened list
    return [node for ring in rings for node in ring]


def connect_rings(
        headlands: List[LinearRing],
        rings: List[List[SecondaryTrackGraphNode]],
        route_params: RoutePlanningConfig) -> None:
    """
    Runs through the given headlands and connects the secondary nodes.
    """

    for i, ring in enumerate(rings):
        # if a cutout ring is located at the edge of the field connect it to the field border (= headland[0])
        to_be_connected = (
            (route_params.working_width > route_params._track_width
                and len(ring) < round(route_params.working_width / route_params._track_width) + route_params._track_width * 0.6
                and headlands[i].distance(headlands[0]) < 3.0 * route_params._track_width)
            or (i > 0 and headlands[i].intersects(headlands[0]))  # Headland crosses field border
        )

        for ii in range(-1, len(ring) - 1):

            node = ring[ii]
            other = ring[ii + 1]

            # if connection is a projection from get_secondary_positions (= no metrics)
            # replace direct connection with neighbors
            if i > 0:
                links = node.edges.copy()
                for connection in links.values():
                    if connection[0].ring_index == 0 and connection[1] is None:

                        metrics = node.calculate_metrics(connection[0].front_secondary, route_params)
                        node.link_node(connection[0].front_secondary, metrics)

                        metrics = node.calculate_metrics(connection[0].back_secondary, route_params)
                        node.link_node(connection[0].back_secondary, metrics)

                        node.remove_link(connection[0])
                        to_be_connected = False

            node.link_front(other, route_params)

            if to_be_connected:
                node.ring_index = -1

        if to_be_connected and len(ring) > 1:
            # connect the ring to the outer headland
            ring.sort(
                key=lambda node: node.position.distance(LinearRing([n.position for n in rings[0]]))
            )
            if len(rings[0]) < 1:
                continue
            outer_nodes = rings[0].copy()
            conn_node_1 = ring[0]
            conn_node_2 = ring[1]

            outer_nodes.sort(key=lambda node: node.distance_to(conn_node_1))
            conn_node_1: SecondaryTrackGraphNode
            neighbor = outer_nodes[0]
            metrics = conn_node_1.calculate_metrics(neighbor, route_params)
            conn_node_1.link_node(neighbor, metrics)

            outer_nodes.sort(key=lambda node: node.distance_to(conn_node_2))
            neighbor = outer_nodes[0]
            metrics = conn_node_2.calculate_metrics(neighbor, route_params)
            conn_node_2.link_node(neighbor, metrics)


def simplify_ab_lines(
        old_ab_lines: List[LineString],
        headlands: List[LinearRing],
        route_params: RoutePlanningConfig) -> List[LineString]:
    """
    Removes any vertices in the ab lines accept for the first and last one.

    Checks every combination of lines if they are split by a bulge in only the inner most headland.
    If they are, connects them into a single ab-line.
    """
    simplify_pairs = set()
    replacements = []
    max_extension_length = 0.0
    difference = route_params._track_width * 0.5
    for ring in headlands:
        max_extension_length += ring.length

    for i in range(len(old_ab_lines) - 1):
        line_1 = old_ab_lines[i]

        if line_1.length < route_params.min_ab_line_length:
            simplify_pairs.add(i)
            if route_params.debug_prints:
                print(f"Removing ab line {line_1} because it is shorter than min_ab_line_length")
            continue

        xl = gt.extend_line(line_1, max_extension_length)

        xl_intersections = []
        for ring in headlands:
            intersection = xl.intersection(ring)
            if isinstance(intersection, MultiPoint):
                xl_intersections.extend(list(intersection.geoms))
            elif isinstance(intersection, Point):
                xl_intersections.append(intersection)
                if route_params.debug_prints:
                    print("Only one intersection with headland for line, might return wrong results")

        # Trim xl to the segment between the two closest intersections
        # This moves the headland intersection check to here and should make it more reliable
        if len(xl_intersections) >= 2:
            centroid = line_1.centroid
            centroid_proj = xl.project(centroid)

            before = [p for p in xl_intersections if xl.project(p) < centroid_proj]
            after = [p for p in xl_intersections if xl.project(p) >= centroid_proj]

            if before and after:
                closest_before = max(before, key=lambda p: xl.project(p))
                closest_after = min(after, key=lambda p: xl.project(p))
                xl_to_headland = gt.substring(xl, xl.project(closest_before), xl.project(closest_after))
            else:
                # centroid is outside the field — fall back to Hausdorff distance over all consecutive segments
                # we might want to remove it again and just demand ab cut to the field border already
                xl_intersections.sort(key=lambda p: xl.project(p))
                best_seg = None
                best_hausdorff = float('inf')
                for k in range(len(xl_intersections) - 1):
                    seg = gt.substring(xl, xl.project(xl_intersections[k]), xl.project(xl_intersections[k + 1]))
                    dist = seg.hausdorff_distance(line_1)
                    if dist < best_hausdorff:
                        best_hausdorff = dist
                        best_seg = seg
                if best_seg is not None:
                    xl_to_headland = best_seg

        elif len(xl_intersections) == 1 and route_params.debug_prints:
            print("Tangente in simplify ab lines.")
        elif xl_intersections == 0:
            simplify_pairs.add(i)
            if route_params.debug_prints:
                print(f"Removing ab line {line_1} because its extension doesn't intersect the headland")
            continue
        best_replacement = None
        best_dist = float('inf')

        if xl_to_headland is None:
            if route_params.debug_prints:
                print("xl to headland is None, this should not happen")
                print(f"Removing ab line {line_1} because its extension doesn't intersect the headland")
                # removes the problematic line instead of crashing. Relevant ab lines might not be worked.
            simplify_pairs.add(i)
            continue

        for ii in range(i + 1, len(old_ab_lines)):
            line_2 = old_ab_lines[ii]

            if line_1.equals(line_2):
                # if they are equal just throw the first away and search replacements for the second
                simplify_pairs.add(i)
                break

            # Find closest point on 2nd line and orient coordinates accordingly
            start = Point(line_2.coords[0])
            end = Point(line_2.coords[-1])
            if line_1.distance(start) < line_1.distance(end):
                target = start
                coords = list(line_2.coords)
            else:
                target = end
                coords = list(line_2.coords)[::-1]

            if difference < xl_to_headland.distance(target):
                continue

            # make constructed line covering the line pair and check for intersection with headland
            if line_2.distance(Point(line_1.coords[0])) < line_2.distance(Point(line_1.coords[-1])):
                coords = list(line_1.coords)[::-1] + coords
            else:
                coords = list(line_1.coords) + coords
            j = 1
            while j < len(coords):
                if Point(coords[j]).distance(Point(coords[j - 1])) < 0.01:
                    coords.pop(j)
                else:
                    j += 1
            test_line = LineString(coords)

            # this check should still be insufficient for some cases
            # but might still be needed for small headland bulges
            intersects_headland = False
            for ring in headlands:
                if test_line.intersects(ring):
                    intersects_headland = True
                    break
            if intersects_headland:
                continue

            distance = line_1.distance(target)
            if distance < best_dist:
                best_dist = distance
                best_replacement = test_line
                best_partner = ii

        if best_replacement:

            simplify_pairs.update([i, best_partner])
            replacements.append(best_replacement)

    new_ab_lines: List[LineString] = old_ab_lines.copy()
    removed_ab_lines = []
    for index in simplify_pairs:
        try:
            removed_ab_lines.append(old_ab_lines[index])
            new_ab_lines.remove(old_ab_lines[index])
        except ValueError:
            continue

    problematic_replacements = []
    # look for overlapping replacements and only take one covering both
    i = 0
    while i < len(replacements):
        replacement = replacements[i]
        ii = i + 1
        while ii < len(replacements):
            other_replacement = replacements[ii]

            if replacement == other_replacement:
                replacements.pop(ii)
            elif replacement.dwithin(other_replacement, 1.0):
                # The following line fails on lines that arent sharing an endpoint
                temp_replacement = linemerge(unary_union([replacement, other_replacement]))
                if isinstance(temp_replacement, MultiLineString):
                    problematic_replacements.append(replacement)
                    problematic_replacements.append(other_replacement)
                    p1, p2 = nearest_points(replacement, other_replacement)

                    # Find which endpoints are closest and connect them
                    if isinstance(replacement, MultiLineString):
                        replacement = linemerge(replacement)
                    if isinstance(other_replacement, MultiLineString):
                        other_replacement = linemerge(other_replacement)
                    coords1 = list(replacement.coords)
                    coords2 = list(other_replacement.coords)

                    # Orient both lines so the closest endpoints face each other
                    if Point(coords1[0]).distance(p1) < Point(coords1[-1]).distance(p1):
                        coords1 = coords1[::-1]
                    if Point(coords2[0]).distance(p2) < Point(coords2[-1]).distance(p2):
                        coords2 = coords2[::-1]

                    temp_replacement = LineString(coords1 + coords2)
                replacement = temp_replacement
                replacements.pop(ii)
            else:
                ii += 1

        i += 1
        if isinstance(replacement, MultiLineString):
            temp_replacement = []
            for line in replacement.geoms:
                temp_replacement.append(line.coords)
            replacement = LineString([coord for line in temp_replacement for coord in line])
        replacement = replacement.simplify(0.001, preserve_topology=True)
        new_ab_lines.append(replacement)

    # discard ab lines if on headland
    multi_headland = MultiLineString(headlands)
    new_ab_lines = [
        line for line in new_ab_lines if (
            line.length > route_params.min_ab_line_length
            and line.centroid.distance(multi_headland) > 0.01
        )
    ]

    return new_ab_lines


def track_system_to_primaries_secondaries(
        track_system: TrackSystem,
        route_params: RoutePlanningConfig
        ) -> Tuple[List[LineString], List[SecondaryTrackGraphNode], List[PlanningMsg]]:
    """
    Returns primary ab_lines, secondary headland_points, warnings.
    """

    warnings = []
    headlands, is_target = get_target_headland_from_track_system(
        track_system,
        route_params,
    )
    if not is_target:
        warnings.append(PlanningMsg.TARGET_HEADLAND_NOT_AVAILABLE)

    ab_lines = list(track_system.ab_lines.geoms)

    ab_lines = simplify_ab_lines(ab_lines, headlands, route_params)
    secondaries = get_secondary_positions(ab_lines, headlands, route_params)

    return ab_lines, secondaries, warnings


def get_track_width(track_system: TrackSystem | List[LineString]) -> float | None:
    """
    Returns the track width of a given track system.
    """
    ab_lines: List[LineString] = []
    if isinstance(track_system, TrackSystem):
        ab_lines = list(track_system.ab_lines.geoms)
    elif isinstance(track_system, list):
        ab_lines = track_system.copy()
    else:
        raise TypeError("track_system must be instance of TrackSystem of list of LineStrings")

    if ab_lines is None:
        return None
    if len(ab_lines) < 2:
        return None

    sort_from_point = ab_lines[0].parallel_offset(track_system.outer_border.exterior.length).centroid
    ab_lines.sort(key = lambda line: line.distance(sort_from_point))

    distances = []
    for i in range(1, len(ab_lines)):
        l1 = ab_lines[i - 1]
        l2 = ab_lines[i]
        distances.append(l1.distance(l2))

    max_count = 0
    margin = 0.0001
    most_common = None

    for i in range(len(distances)):
        current_value = distances[i]
        count = 0

        for ii in range(len(distances)):
            if abs(current_value - distances[ii]) <= margin:
                count += 1

        if count > max_count:
            max_count = count
            most_common = current_value

    track_width = most_common

    if abs(track_width - 14.0) < 0.001:
        track_width = 14.0
    elif abs(track_width - 13.716) < 0.001:
        track_width = 13.716

    return track_width


def get_inner_border_from_track_system(track_system: TrackSystem, route_params: RoutePlanningConfig) -> MultiPolygon | None:
    """
    Generates the inner border on a track system.
    """

    headland_prep: list[float | None] = []
    if route_params.working_width is None:
        return None
    mh = route_params._track_width / 2
    new_ww = (
        route_params.working_width - (route_params.working_width % (mh))
        if ((route_params.working_width % (mh)) <= mh / 2)
        else (route_params.working_width - (route_params.working_width % (mh)) + (mh))
    )
    headland_prep = _get_headland_prep(track_system, route_params)
    if route_params.last_driven_headland_index is not None and route_params.last_driven_headland_index < len(
        track_system.headlands
    ):
        print(f"Using last driven headland index {route_params.last_driven_headland_index} for inner border calculation")
        outer_headland_rings = track_system.headlands[route_params.last_driven_headland_index]
        outer_headland_poly = Polygon(outer_headland_rings[0], outer_headland_rings[1:])
        inner_border = outer_headland_poly.buffer(route_params.working_width / 2 * -1, resolution=20, cap_style=2, join_style=2)
    else:
        offs = 0
        if sum(headland_prep) < new_ww:
            offs += new_ww - sum(headland_prep)

        rest = (sum(headland_prep)) % new_ww
        if rest > 0 and sum(headland_prep) > (new_ww):
            snip = headland_prep.pop()
            temp = snip + (new_ww - rest)
            if temp > new_ww / 2:
                offs = new_ww / 2
            else:
                offs = temp
        outer_headland_rings = track_system.headlands[0]  # TODO make this support multi part fields
        outer_headland_poly = Polygon(outer_headland_rings[0], outer_headland_rings[1:])
        inner_border = outer_headland_poly.buffer((sum(headland_prep[1:]) + offs) * -1, resolution=40, cap_style=2, join_style=2)

    if isinstance(inner_border, Polygon):
        inner_border = MultiPolygon([inner_border])
    return inner_border


def _get_headland_prep(track_system: TrackSystem, route_params: RoutePlanningConfig) -> list[float | None]:
    headland_prep = []
    for i, value in enumerate(track_system.headland_config):
        head_width = value * route_params._track_width
        if i == 0:
            head_width /= 2
        headland_prep.append(head_width)
    headland_prep.append(0.5 * route_params._track_width)
    return headland_prep


def get_headland_index_of_path_on_track_system(path: LineString | MultiLineString, track_system: TrackSystem) -> int | None:
    """
    Looks for the index of the headland of a track system with the most intersection with a given path over that track system.
    """

    path_buffered = path.buffer(0.1)

    multi_headland_geoms = [
        MultiLineString(headland) for headland in track_system.headlands
    ]

    def headland_intersection_length(headland_index: int) -> float:
        multi_headland_geom = multi_headland_geoms[headland_index]
        intersection = path_buffered.intersection(multi_headland_geom)

        multi_line_intersection: MultiLineString | None = None

        if isinstance(intersection, GeometryCollection):
            lines: List[LineString] = []
            for geom in intersection.geoms:
                if isinstance(geom, LineString):
                    lines.append(geom)
                elif isinstance(geom, MultiLineString):
                    lines.extend(geom.geoms)
            multi_line_intersection = MultiLineString(lines)

        elif isinstance(intersection, MultiLineString):
            multi_line_intersection = intersection

        if multi_line_intersection is None:
            return 0.0

        return multi_line_intersection.length

    all_headland_indices = list(range(len(track_system.headlands)))
    target_index = max(all_headland_indices, key = headland_intersection_length)

    if multi_headland_geoms[target_index].distance(path) > 0.0:
        # path is not actually near the given field
        return None

    return target_index


def get_ab_lines_on_path(track_system: TrackSystem, path: LineString, max_distance: float = 1.0) -> List[LineString]:
    """
    Returns the list of ab lines that are included in the given path within 'max_distance'.
    """
    ab_lines = list(track_system.ab_lines.geoms)
    parallel_path_segments = []
    test_line = ab_lines[0]
    for i in range(1, len(path.coords)):
        p1 = Point(path.coords[i - 1])
        p2 = Point(path.coords[i])
        segment = LineString([p1, p2])
        angle = abs(gt.angle_between_lines(test_line, segment))
        if (angle < 0.001 or abs(angle - pi) < 0.001):
            parallel_path_segments.append(segment)

    parallel_path = MultiLineString(parallel_path_segments)

    included_ab_lines = []
    for ab_line in ab_lines:
        if ab_line.dwithin(parallel_path, max_distance):
            included_ab_lines.append(ab_line)
    if len(included_ab_lines) == 0:
        return None

    return included_ab_lines


def get_working_width_of_path_on_track_system(
        path_input: LineString | MultiLineString,
        track_system: TrackSystem) -> float | None:
    """
    Returns the working width of a path on a track system.
    """

    if isinstance(path_input, MultiLineString):
        coords = []
        [coords.extend(segment.coords) for segment in path_input.geoms]
        path = LineString(coords)
    elif isinstance(path_input, LineString):
        path = path_input
    else:
        raise TypeError("Path has to be LineString or MultiLineString")

    ab_lines_list = get_ab_lines_on_path(track_system, path)

    dists_path = []
    for ab_line in ab_lines_list:
        candidates = list(ab_lines_list)
        candidates.sort(key=lambda x: x.distance(ab_line))
        if len(candidates) < 3:
            continue

        nearest = candidates[1]
        second_nearest = candidates[2]
        dists_path.append(nearest.distance(ab_line))
        dists_path.append(second_nearest.distance(ab_line))

    return np.median(dists_path)


def working_corridor_of_ab_line(
        ab_line: LineString,
        working_width: float,
        border_poly: Polygon,
        turning_headland: LinearRing,
        cutout_polys: List[Polygon] | None,
        start_point: Point | None = None) -> Polygon | None:
    """
    Get the working corridor of a line with exactly 2 points.

    The working corridor is the area that is reachable by the robot.

    Parameters
    ----------
    ab_line : LineString
        The line for which the working corridor should be calculated.
    working_width : float
        The width of the working corridor.
    border_poly : Polygon
        The border of the field.
    cutout_polys : List[Polygon]
        The cutouts of the field.
    start_point : Point | None
        The point where the robot is located.

    Returns
    -------

    The working corridor as a Polygon or None if the robot is not contained in the working corridor.
    """
    if ab_line is None:
        return None
    if working_width is None:
        return None
    if border_poly is None:
        return None
    if cutout_polys is None:
        cutout_polys = []
    if start_point is None:
        start_point = ab_line.centroid

    try:
        parallel_distance = working_width / 2.0

        parallel_buffer = ab_line.buffer(
            parallel_distance, cap_style='square', join_style='round'
        )

        inter_buffer = border_poly.intersection(parallel_buffer)

        for polygon in cutout_polys:
            if inter_buffer.intersects(polygon):
                inter_buffer = inter_buffer.difference(polygon)

        # is Multi or Polygon?
        start_point_contained = False
        if isinstance(inter_buffer, MultiPolygon):

            for buffer in inter_buffer.geoms:
                buffer = oriented_envelope(buffer)
                if start_point and buffer.contains(start_point):
                    inter_buffer = buffer
                    start_point_contained = True
                    break

        else:
            inter_buffer = oriented_envelope(inter_buffer)
            if inter_buffer.contains(start_point):
                start_point_contained = True

        if start_point and not start_point_contained:
            return None

        return inter_buffer

    except Exception:
        return None


def get_corridor_line(
    ab_line: LineString,
    working_width: float,
    turning_headland: LinearRing,
    inner_border: LinearRing | Polygon | MultiPolygon,
    implement_working_offset: float = 0.0
) -> LineString | None:
    """Returns the working corridor line of an ab line within a turning headland.

    Calculates the working corridor line. This respects the working width and the turning headland geometry.
    """
    headland_poly = Polygon(turning_headland)
    min_width = 0.0001
    half_width = working_width / 2.0 - min_width

    # Resolve MultiPolygon to a single Polygon covering all parts near the ab_line
    if isinstance(inner_border, MultiPolygon):
        touching = [poly for poly in inner_border.geoms if poly.distance(ab_line) < 1e-4]
        if len(touching) > 1:
            inner_border = unary_union(touching).convex_hull
        elif len(touching) == 1:
            inner_border = touching[0]
        else:
            inner_border = min(inner_border.geoms, key=lambda poly: poly.distance(ab_line))

    inner_border_poly = inner_border if isinstance(inner_border, Polygon) else Polygon(inner_border)

    def get_intersection_line_savely(line: LineString, polygon: Polygon) -> LineString | None:
        """Gets the intersection between a line and a polygon and ensure it returns a LineString."""
        intersection = line.intersection(polygon, grid_size=0)
        if isinstance(intersection, MultiLineString):
            # Return the longest segment to avoid picking a spurious short fragment
            candidates = [g for g in intersection.geoms if g.length > 0.1]
            if candidates:
                return max(candidates, key=lambda g: g.length)
        elif isinstance(intersection, LineString):
            return intersection
        elif isinstance(intersection, GeometryCollection):
            candidates = [g for g in intersection.geoms if isinstance(g, LineString) and g.length > 0.1]
            if candidates:
                return max(candidates, key=lambda g: g.length)
        else:
            return None

    # Extend AB line to the turning headland
    # But at first handle the case, where the turning headland is at a cutout

    extended = get_intersection_line_savely(ab_line, headland_poly)
    if extended is None or extended.is_empty:
        extended = gt.extend_line_in_bounds(ab_line, inner_border_poly, inner_border.length)
        extended = get_intersection_line_savely(extended, inner_border_poly)
    else:
        extended = gt.extend_line_in_bounds(ab_line, headland_poly, headland_poly.length)
        extended = get_intersection_line_savely(extended, headland_poly)
    if extended is None or extended.is_empty:
        # this can happen when we are working an ab line which only has working area inside the inner_border
        extended = ab_line
    # Create left/right offsets and clip to headland
    left_offset = extended.offset_curve(half_width)
    right_offset = extended.offset_curve(-half_width)

    left_clipped = left_offset.intersection(inner_border_poly)
    right_clipped = right_offset.intersection(inner_border_poly)
    middle_clipped = extended.intersection(inner_border_poly)

    def fallback_line(offset_line: LineString) -> LineString:
        """Handels the case where an offset line is completely outside the inner border and thus the intersection is empty.

        This basically handels the edge of the field. Using nearest points is a bit dodgy but it works. Could lead to unnecessary
        hooks.
        """
        p1, _ = gt.nearest_points(inner_border_poly, offset_line.boundary.geoms[0])
        p2, _ = gt.nearest_points(inner_border_poly, offset_line.boundary.geoms[-1])
        if p1.distance(p2) < 0.01:
            # trick to avoid errors when both points are the same
            p1 = extended.boundary.geoms[-1]
            p2 = extended.boundary.geoms[0]
        return LineString([p1, p2])

    if left_clipped is None or left_clipped.is_empty:
        left_clipped = fallback_line(left_offset)
    if right_clipped is None or right_clipped.is_empty:
        right_clipped = fallback_line(right_offset)
    if middle_clipped is None or middle_clipped.is_empty:
        middle_clipped = fallback_line(extended)

    # TODO find a good check to filter unnecessary hooks at cutouts
    def recombine_multilinestring(ml_string: MultiLineString) -> LineString:
        new_line = []
        for seg in ml_string.geoms:
            if seg.length < 0.1:
                continue
            new_line.extend(seg.coords)
        if len(new_line) < 2:
            return extended.reverse()  # trick to avoid errors
        return LineString(new_line)

    if isinstance(left_clipped, MultiLineString):
        left_clipped = recombine_multilinestring(left_clipped)
    if isinstance(right_clipped, MultiLineString):
        right_clipped = recombine_multilinestring(right_clipped)
    if isinstance(middle_clipped, MultiLineString):
        middle_clipped = recombine_multilinestring(middle_clipped)
    # mind the working offset of the implement
    left_clipped = gt.extend_line(left_clipped, implement_working_offset, extend_back=False)
    right_clipped = gt.extend_line(right_clipped, implement_working_offset, extend_back=False)
    middle_clipped = gt.extend_line(middle_clipped, implement_working_offset, extend_back=False)
    # comment this out, when we can make sure that the driver follows the path in the right direction
    # left_clipped = gt.substring(left_clipped, implement_working_offset, left_clipped.length)
    # right_clipped = gt.substring(right_clipped, implement_working_offset, right_clipped.length)
    # middle_clipped = gt.substring(middle_clipped, implement_working_offset, middle_clipped.length)

    left_start = extended.project(left_clipped.boundary.geoms[0])
    left_end = extended.project(left_clipped.boundary.geoms[-1])
    if left_end <= left_start:
        left_start, left_end = left_end, left_start

    right_start = extended.project(right_clipped.boundary.geoms[0])
    right_end = extended.project(right_clipped.boundary.geoms[-1])
    if right_end <= right_start:
        right_start, right_end = right_end, right_start

    middle_start = extended.project(middle_clipped.boundary.geoms[0])
    middle_end = extended.project(middle_clipped.boundary.geoms[-1])
    if middle_end <= middle_start:
        middle_start, middle_end = middle_end, middle_start

    valid_start = min(left_start, right_start, middle_start)
    valid_end = max(left_end, right_end, middle_end)

    # if this is the case, we couldn't find a corridor that we have to work or is smaller than 0.1
    if valid_end <= valid_start:
        valid_start = extended.length / 2 - 0.01
        valid_end = extended.length / 2 + 0.01

    corridor_line = gt.substring(extended, valid_start, valid_end)
    if isinstance(corridor_line, LineString) and not corridor_line.is_empty:
        return remove_repeated_points(corridor_line, 0.01)
    return None  # remove_repeated_points(ab_line, 0.01)
