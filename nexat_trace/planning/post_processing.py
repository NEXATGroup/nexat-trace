
from dataclasses import dataclass
from typing import List, Tuple

from shapely import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    minimum_bounding_radius,
    remove_repeated_points,
    simplify,
)
from shapely.ops import split, substring

from nexat_trace.planning import curve_calculation
from nexat_trace.planning.curve_calculation import insert_hook_stops_to_ab, insert_hook_stops_to_headland
from nexat_trace.planning.route import Route
from nexat_trace.shared.config import CorridorStrategy, PostSteps, RoutePlanningConfig
from nexat_trace.shared.exceptions import RoutePlanningError
from nexat_trace.util import geom_tools
from nexat_trace.util.field_conversion import get_corridor_line

"""
This module defines post processing steps for routes

Defined functions must take only the Route as a parameter and
act in place.
The RoutePlanner automatically looks up all available steps
and their corresponding functions in the FUNCTIONS dictionary.
"""


@dataclass
class Intersect:
    """Class holding relevant data of the intersection."""

    start: Point
    end: Point
    segment: LinearRing
    centroid: Point


def cutout_avoidance(route: Route) -> None:
    """
    Checks a given Route if the path intersects with any specialized cutout segments and changes the path accordingly.
    """

    original_path = LineString(route._path)
    cutout_segments = route._track_system.obstacle_avoidance_segments

    # is there anything to do here?
    if not any(original_path.intersects(segment) for segment in cutout_segments):
        return

    current_path = route._path.copy()

    if not all(isinstance(segment, (LinearRing, LineString)) for segment in cutout_segments):
        raise TypeError("not all obstacle_avoidance_segments were of types LineString or LinearRing")

    all_intersections: List[Intersect] = []
    start_end: List[Point] = []

    # Find intersections between cutout segments and ab lines
    all_intersections, start_end = find_intersections(
        original_path, cutout_segments, route._track_system.outer_border, route._route_params.working_width
        )

    current_path = cut_to_start_end(start_end, current_path, original_path)

    all_intersections.sort(key=lambda intersect: original_path.project(intersect.start))
    final_path: List[Point] = [current_path.pop(0)]

    last_intersection: Intersect | None = None

    # Morph Intersections into curves and sort into final path
    while current_path:

        # Free To Go or points on path till next obstacle
        if (not all_intersections
                or original_path.project(Point(current_path[0])) < original_path.project(all_intersections[0].start)):
            final_path.append(current_path.pop(0))

        else:
            # Find way around obstacle
            intersection = all_intersections.pop(0)

            # Points covered by a curve
            while original_path.project(Point(current_path[0])) <= original_path.project(intersection.end):
                current_path.pop(0)

            aligned_segment = geom_tools.ring_with_origin_at(intersection.segment, intersection.start)
            segment_cut = geom_tools.get_substring_on_linearring(
                aligned_segment,
                intersection.start,
                intersection.end
            )

            aligned_segment_reversed = aligned_segment.reverse()
            segment_cut_reversed = geom_tools.get_substring_on_linearring(
                aligned_segment_reversed,
                intersection.start,
                intersection.end
            )

            border_collision_on_full_circle = (
                route._field_border.dwithin(intersection.segment, route._route_params.working_width / 2)
            )

            if border_collision_on_full_circle:
                if route._field_border.dwithin(segment_cut, route._route_params.working_width / 2):
                    aligned_segment = aligned_segment_reversed
                    segment_cut = segment_cut_reversed

            elif aligned_segment.project(intersection.end) > aligned_segment_reversed.project(intersection.end):
                aligned_segment = aligned_segment_reversed
                segment_cut = segment_cut_reversed

            coords = [final_path.pop(-1), intersection.start]

            # extend run up segment backwards
            while (len(final_path) > 0
                    and Point(coords[0]).distance(intersection.segment) < route._route_params.vehicle_turning_radius * 2
                    and not (last_intersection is not None
                             and original_path.project(last_intersection.end)
                             > original_path.project(final_path[-1]) - route._route_params._track_width * 0.35)):
                coords.insert(0, final_path.pop(-1))

            curve_on_ab_segment = LineString(coords)
            # cut back
            while curve_on_ab_segment.distance(intersection.segment) < 1.0:
                curve_on_ab_segment = substring(curve_on_ab_segment, 0.0, curve_on_ab_segment.length - 0.33)

            # remove numerical artefacts
            curve_on_ab_segment = remove_repeated_points(curve_on_ab_segment, 0.001)
            segment_cut = remove_repeated_points(segment_cut, 0.001)
            # curve onto segment
            curve_on = curve_calculation.get_simple_turn_to_headland(
                curve_on_ab_segment,
                segment_cut,
                aligned_segment,
                route._field_border,
                route._route_params
            )

            if route._route_params.debug_prints and curve_on is None:
                print("Could not find curve onto cutout segment on ab line")
                continue

            if curve_on.distance(intersection.segment) > 0.0001:
                # extend curve onto segment
                extension = geom_tools.extend_line(
                    LineString(
                        [
                            curve_on.coords[-2],
                            curve_on.coords[-1]
                        ]
                    ),
                    route._field_border.length
                )
                extension_intersection = extension.intersection(intersection.segment)
                if isinstance(extension_intersection, MultiPoint):
                    extension_points = list(extension_intersection.geoms)
                    extension_point = min(extension_points, key=lambda p: extension.project(p))
                else:
                    extension_point, _ = geom_tools.nearest_points(intersection.segment, extension)

                curve_on = LineString(list(curve_on.coords))
            s_d = route._route_params._segmentation_avoidance_distance  # s_d small distance to prevent unnecessary segmentation
            end_dist = curve_on.length - s_d
            curve_on = substring(
                curve_on,
                max(0, min(s_d, end_dist - 0.001)),
                end_dist
            )

            end_dist = curve_on_ab_segment.project(Point(curve_on.coords[0])) - s_d
            curve_on_ab_segment = substring(
                curve_on_ab_segment,
                max(0, min(s_d, end_dist - 0.001)),
                end_dist,
            )

            coords = [intersection.end]

            # extend segment away from cutout segment to plan curve off of the segment
            interpolation_cnt = 1
            while (len(current_path) > 0
                    and (len(coords) < 2
                         or coords[-1].distance(intersection.segment) < route._route_params.vehicle_turning_radius * 2)):
                # if there is still intersections left dont extend curve off segment beyond next intersection
                if len(all_intersections) > 0:
                    next_intersection_projection_length = original_path.project(all_intersections[0].start)
                    # is next path point ok?
                    if original_path.project(current_path[0]) < next_intersection_projection_length:
                        coords.append(current_path.pop(0))

                    else:
                        interpolation_length = original_path.project(coords[-1]) + interpolation_cnt * 1.0
                        if interpolation_length < next_intersection_projection_length:
                            coords.append(original_path.interpolate(interpolation_length))
                            interpolation_cnt += 1
                        else:
                            break  # not good if reaches here

                else:
                    coords.append(current_path.pop(0))

            rest_path = LineString(coords)

            # curve off of the segment
            curve_off = curve_calculation.get_simple_turn_to_ab(
                rest_path,
                segment_cut,
                aligned_segment,
                route._field_border,
                route._route_params
            )

            end_dist = curve_off.length
            curve_off = substring(
                curve_off,
                max(0, min(s_d, end_dist - 0.001)),
                end_dist
            )
            if route._route_params.debug_prints and curve_off is None:
                print("Could not find curve off of cutout segment on ab line")
                continue

            rest_path = substring(
                rest_path,
                max(0, min(rest_path.project(Point(curve_off.coords[-1])) + s_d, rest_path.length - 0.001)),
                rest_path.length,
            )

            # path on segment
            cut_segment_path = geom_tools.get_substring_on_linearring(
                aligned_segment,
                Point(curve_on.coords[-1]),
                Point(curve_off.coords[0])
            )
            end_dist = cut_segment_path.length - s_d
            cut_segment_path = geom_tools.substring(
                cut_segment_path, max(0, min(s_d, end_dist - 0.001)), end_dist
                )

            should_be_circled = (
                route._route_params.fully_circle_cutouts  # is parameter set?
                and not LineString(final_path).dwithin(aligned_segment, 0.1)  # already visited?
                and not border_collision_on_full_circle
            )

            cut_segment_path_points = [Point(coord) for coord in cut_segment_path.coords]
            interpolated_on_ring = geom_tools.extend_line(
                curve_on, route._route_params.vehicle_turning_radius * 0.5
            ).intersection(aligned_segment)
            if interpolated_on_ring.is_empty:
                interpolated_on_ring = geom_tools.nearest_points(geom_tools.extend_line(
                    curve_on, route._route_params.vehicle_turning_radius * 0.5
                ),
                 aligned_segment
                )[1]
            if isinstance(interpolated_on_ring, MultiPoint):
                interpolated_on_ring = list(interpolated_on_ring.geoms)[0]
            if should_be_circled:
                aligned_segment_circle = geom_tools.ring_with_origin_at(aligned_segment, interpolated_on_ring)
                # insert circle points before the rest of the calculated segment
                circle = LineString([Point(coord) for coord in aligned_segment_circle.coords])
                projected_end = circle.project(cut_segment_path.boundary.geoms[0])
                end_dist = max(0, min(circle.length - s_d, projected_end - s_d))
                circle = geom_tools.substring(circle, max(0, min(s_d, end_dist - 0.001)), end_dist)
                cut_segment_path_points = (
                    [Point(coord) for coord in list(circle.coords)]
                    + cut_segment_path_points
                )

            segment_path_points = LineString(
                [Point(coord) for coord in curve_on_ab_segment.coords]
                + [Point(coord) for coord in curve_on.coords]
                + cut_segment_path_points
                + [Point(coord) for coord in curve_off.coords]
                + [Point(coord) for coord in rest_path.coords]
            )
            segment_path_corrds = list(simplify(segment_path_points, 5e-4).coords)
            segment_path_points = [Point(coord) for coord in segment_path_corrds]

            final_path.extend(segment_path_points)
            last_intersection = intersection
    _rebuild_line(route, final_path)


def evade_rois(route: Route):
    """Calculates the evasion maneuvers araound an obstacle and insert them into the path."""

    original_path = LineString(route._path)
    cutout_segments = route._track_system.to_be_evaded_obstacles
    cutout_segments_buffered = [
        cutout.buffer(route._route_params.working_width / 2, resolution = 40).exterior for cutout in cutout_segments]

    # is there anything to do here?
    if not any(original_path.intersects(segment) for segment in cutout_segments_buffered):
        return

    current_path = route._path.copy()

    if not all(isinstance(segment, (LinearRing, LineString)) for segment in cutout_segments):
        raise TypeError("not all obstacle_avoidance_segments were of types LineString or LinearRing")

    all_intersections: List[Intersect] = []
    start_end: List[Point] = []

    # Find intersections between cutout segments and ab lines
    all_intersections, start_end = find_intersections(
        original_path, cutout_segments_buffered, route._track_system.outer_border, route._route_params.working_width
        )

    current_path = cut_to_start_end(start_end, current_path, original_path)

    all_intersections.sort(key=lambda intersect: original_path.project(intersect.start))
    final_path: List[Point] = [current_path.pop(0)]

    # Morph Intersections into curves and sort into final path
    while current_path:

        # Free To Go or points on path till next obstacle
        if (not all_intersections
                or original_path.project(Point(current_path[0])) < original_path.project(all_intersections[0].start)):
            final_path.append(current_path.pop(0))
        else:
            # we need to curve around the obstacle
            # but first first give space for the start point of the maneuver
            while (
                original_path.project(final_path[-1])
                > max(original_path.project(all_intersections[0].start) - route._route_params.vehicle_turning_radius, 0)
            ):
                final_path.pop()
            # the path around
            obstacle = min(cutout_segments, key=lambda seg: all_intersections[0].centroid.distance(seg))
            path_project_afer = (
                original_path.project(all_intersections[0].end)
                + route._route_params.working_width
                + route._route_params.vehicle_turning_radius
            )
            start_segment = geom_tools.substring(original_path, original_path.project(final_path[-1]), path_project_afer + 0.1)
            current_point = final_path[-1] if len(final_path) > 0 else Point(start_segment.coords[0])

            bounding_ab_lines = determine_bounding_abs(
                obstacle,
                route._track_system.ab_lines,
                route._route_params.working_width,
                route._route_params._track_width)

            path_around_obs = calculate_obstacle_evasion(
                start_segment,
                current_point,
                route._route_params.working_width,
                obstacle,
                route._route_params.debug_prints,
                bounding_ab_lines,
                route._route_params.vehicle_turning_radius,
                route._track_system.outer_border.exterior,
            )

            # TODO implement for only drive all?
            # route._route_params.corridor_strategy = CorridorStrategy.DRIVE_ALL.value
            if path_around_obs is not None and route._route_params.corridor_strategy != CorridorStrategy.DRIVE_NONE.value:
                inner_border = route._full_inner_border
                start_point_ab_candidate = min(
                    ((ab, ab.distance(all_intersections[0].start))for ab in route._track_system.ab_lines.geoms),
                    key=lambda t: t[1]
                    )
                end_point_ab_candidate = min(
                    ((ab, ab.distance(all_intersections[0].end))for ab in route._track_system.ab_lines.geoms),
                    key=lambda t: t[1]
                    )
                ab_line = min([start_point_ab_candidate, end_point_ab_candidate], key=lambda ab: ab[1])[0]
                outer_turning_head = route._target_headlands[0]
                path_with_hooks = insert_hooks_around_obs(
                    path_around_obs,
                    obstacle,
                    inner_border,
                    outer_turning_head,
                    ab_line,
                    route._route_params,
                    route._track_system.outer_border
                    )
                path_around_obs = path_with_hooks if path_with_hooks is not None else path_around_obs
            if path_around_obs is None:
                # this should only be at start and end and if we don't have enough space
                if len(final_path) == 0:
                    final_path.append(all_intersections[0].end)
                elif original_path.project(all_intersections[0].end) - original_path.length < 5:
                    final_path.append(all_intersections[0].start)
            else:
                final_path.extend([Point(coord) for coord in path_around_obs.coords])

            while len(current_path) > 0 and original_path.project(final_path[-1]) > original_path.project(current_path[0]):
                current_path.pop(0)

            all_intersections.pop(0)
    _rebuild_line(route, final_path)


def find_intersections(path: LineString,
                       obstacles: List[Polygon] | List[LinearRing],
                       field_border: LinearRing,
                       working_width: float = 14.0) -> Tuple[List[Intersect], List[Point]]:
    """Finds the intersections of the path with the obstacles.

    Returns the info about the intersection as an Intersect object. The centroid is used to identify the obstacle
    """
    all_intersections: List[Intersect] = []
    start_end: List[Intersect] = []

    for obstacle_orig in obstacles:
        obstacle = obstacle_orig.exterior if isinstance(obstacle_orig, Polygon) else obstacle_orig
        # Get the centroid for obstacle identification
        if hasattr(obstacle, 'centroid'):
            centroid = obstacle.centroid
        else:
            centroid = obstacle

        intersections = path.intersection(obstacle)

        # Intersections
        if isinstance(intersections, (MultiPoint, LineString)) and intersections.is_empty:
            continue

        # Handle different intersection types
        intersection_points: List[Point] = []

        if isinstance(intersections, LineString):
            # LineString intersection: use start and end points
            intersection_points = [
                Point(intersections.coords[0]),
                Point(intersections.coords[-1]),
            ]
        elif isinstance(intersections, MultiPoint):
            # Multiple points: convert to list and sort by projection on path
            intersection_points = list(intersections.geoms)
            intersection_points.sort(key=lambda p: path.project(p))
            if len(intersection_points) % 2 != 0:
                # ungerade nur im Fall unwahrscheinlichen Fall einer Tangente oder wenn Start oder Stop im obstacle liegt.
                poly_obstacle = Polygon(obstacle) if isinstance(obstacle, LinearRing) else obstacle
                if poly_obstacle.contains(path.boundary.geoms[0]):
                    start_end.append(intersection_points.pop(0))
                elif poly_obstacle.contains(path.boundary.geoms[-1]):
                    start_end.append(intersection_points.pop(-1))
                # when the path starts in an obstacle it also should end in one for even numer of intersection points
                no_left_over_half_crosses = True
                no_left_over_half_crosses = no_left_over_half_crosses and not poly_obstacle.contains(path.boundary.geoms[0])
                no_left_over_half_crosses = no_left_over_half_crosses and not poly_obstacle.contains(path.boundary.geoms[-1])
                if poly_obstacle.contains(path.boundary.geoms[0]) or poly_obstacle.contains(path.boundary.geoms[-1]):
                    print("Intersections are ordered unexpectedly")
                    raise RoutePlanningError("Intersections are ordered unexpectedly.")
            if (len(intersection_points) > 4 and
                path.project(Point(intersection_points[-1]))
                - path.project(Point(intersection_points[0]))
                < 4 * working_width  # Circle circumference approximation
                    and field_border.dwithin(obstacle, working_width / 2)):
                all_intersections.append(Intersect(intersection_points[0], intersection_points[-1], obstacle, centroid))
                continue
        elif isinstance(intersections, Point):
            start_end.append(intersections)

        # Pair up intersection points (start and end of each intersection segment)
        for i in range(0, len(intersection_points) - 1, 2):
            if i + 1 < len(intersection_points):
                all_intersections.append(Intersect(
                    start=intersection_points[i],
                    end=intersection_points[i + 1],
                    segment=obstacle,
                    centroid=centroid
                ))

    return all_intersections, start_end


def cut_to_start_end(start_end: List[Point], current_path: LineString, original_path: LineString) -> LineString:
    """Cuts the current path to start at the first intersection and end at the last.

    This does only make sense, if the start or end of the original path lies within an obstacle
    """
    while start_end:

        intersection = start_end.pop(0)
        dist = original_path.project(intersection)

        if dist > 0.5:
            while original_path.project(current_path[-1]) > dist:
                current_path.pop(-1)
            current_path.append(intersection)

        else:
            while original_path.project(Point(current_path[0])) < dist:
                current_path.pop(0)
            current_path.insert(0, intersection)

    return current_path


def calculate_obstacle_evasion(
        segment: LineString,
        current_point: Point,
        working_width: float,
        obstacle: Polygon,
        debug_prints: bool,
        bounding_ablines: LineString | MultiLineString,
        turning_radius: float,
        bounds: LinearRing,
) -> LineString:
    """Calculate an evasion path around an obstacle using a circular arc with clearance.

    Generates a smooth detour around the given obstacle by:
    1. Projecting the obstacle centroid onto the original segment
    2. Creating a circular arc with buffer offset for working width and cabin clearance
    3. Connecting start → arc → target using the shorter arc path
    4. Optionally using AB-lines as the evasion path if the arc crosses them
    5. Smoothing connections with Dubins curves

    Args:
        segment: The original path segment to detour from.
        current_point: Current vehicle position (used for Dubins curve generation).
        working_width: Vehicle working width in meters.
        obstacle: Polygon obstacle to evade.
        debug_prints: If True, print debug messages for troubleshooting.
        bounding_ablines: AB-lines that may be used as alternative evasion paths.
        turning_radius: Vehicle turning radius for Dubins curve generation.

    Returns
    -------
        LineString of the complete evasion path (start_segment → arc → target_segment),
        or None if evasion calculation fails.
    """
    # TODO decide whether to put this into geom_tools, curve_calculation or leave it here
    # TODO derive these from route_params
    # Split segment at the obstacle projection point
    obstacle_centroid = obstacle.centroid
    split_distance = segment.project(obstacle_centroid)
    # Ensure we don't split at the very start or end
    split_distance = max(1.0, min(split_distance, segment.length - 1.0))

    # Use width as split offset so the arc endpoints land exactly on
    # start_segment end / target_segment start
    mbr = minimum_bounding_radius(obstacle)
    split_offset = max(0.5, mbr)
    start_segment = substring(segment, 0, split_distance - split_offset)
    target_segment = substring(segment, split_distance + split_offset, segment.length)

    if start_segment is None or start_segment.is_empty or target_segment is None or target_segment.is_empty:
        print(
            f"calculate_path_around_obstacle: segment split failed (split_dist={split_distance:.1f},"
            + f"seg_len={segment.length:.1f})"
            )
        return None

    # Split the circle arc using the original segment as cutting line
    obstacle_buffered = obstacle.buffer(working_width / 2, resolution=40).exterior
    parts = split(LineString(obstacle_buffered), segment).geoms
    if len(parts) == 3:
        parts = [parts[1], LineString(list(parts[-1].coords) + list(parts[0].coords))]

    best_short_path = None
    # TODO collision check with outer border
    # TODO more than two intersections <--- well in theory that just be handled by the amount of intersections
    # TODO Debug LineString split into two for no apparent reason.
    if len(parts) >= 2:
        # Pick the shorter arc – it goes around the near side of the obstacle. <--- for straight lines
        candidate = min(parts, key=lambda p: p.length)
        # Orient so it goes from start_segment end → target_segment start
        if candidate.boundary.geoms[0].distance(
                start_segment.boundary.geoms[-1]) > candidate.boundary.geoms[-1].distance(start_segment.boundary.geoms[-1]):
            candidate = candidate.reverse()
        best_short_path = candidate

    if best_short_path is None and debug_prints:
        print("calculate_path_around_obstacle: Could not split circle into two parts on either side")
        return None
    short_path = best_short_path

    # this may cause trouble with hooks
    if bounding_ablines is not None and bounding_ablines.intersects(short_path):
        if debug_prints:
            print("Evasion move crosses AB-line, use this as obstacle path!")
        ab_path = (bounding_ablines if isinstance(bounding_ablines, LineString)
                   else min(bounding_ablines.geoms, key=lambda ab_line: ab_line.distance(short_path))
                   )
        intersections = short_path.intersection(ab_path)
        if isinstance(intersections, MultiPoint) and len(intersections.geoms) == 2:
            start_point = min(intersections.geoms, key=lambda point: short_path.project(point))
            end_point = max(intersections.geoms, key=lambda point: short_path.project(point))
            ab_path_substring = substring(ab_path, ab_path.project(start_point), ab_path.project(end_point))

            # Validate the extracted AB-path segment has viable geometry for evasion
            # Check: 1) minimum length to maneuver, 2) proximity only to the extracted segment (not entire AB line)
            min_maneuver_length = working_width * 0.5
            if (ab_path_substring is None or ab_path_substring.is_empty or
                ab_path_substring.length < min_maneuver_length or
                    segment.dwithin(ab_path_substring, 1)):  # Check only the portion we'll use
                return None

            ab_path = ab_path_substring
            start_arc_segment = substring(short_path, 0, short_path.project(start_point))
            end_arc_segment = substring(short_path, short_path.project(end_point, True), 1, True)
            gen_path_1 = None
            gen_path_2 = None
            gen_path_1, _, _ = geom_tools.get_curve(
                current_line_oriented=start_arc_segment,
                target_line=ab_path,
                turning_radius=turning_radius,
                step_size=32.0,
                robot_point=None,
                border_buffer=None,
                extend_start_line=True,
                extend_target_line=True)
            gen_path_2, _, _ = geom_tools.get_curve(
                ab_path,
                end_arc_segment,
                turning_radius,
                32.0,
                None,
                None,
                extend_start_line=True,
                extend_target_line=True)

            short_path = geom_tools.combine_paths(current_path=start_arc_segment, next_paths=[gen_path_1, ab_path])
            short_path = geom_tools.combine_paths(current_path=short_path, next_paths=[gen_path_2, end_arc_segment])
        else:
            return None
    gen_path, smoothed_evasion = assemble_path_around_obstacle(
        start_segment=start_segment,
        obstacle_segment=short_path,
        target_segment=target_segment,
        current_point=None,
        turning_radius=turning_radius,
        bounds=bounds
        )
    if gen_path is None:
        if debug_prints:
            print("calculate_path_around_obstacle: assemble_path_around_obstacle returned None")
        return None

    return gen_path if gen_path is not None else None


def assemble_path_around_obstacle(
        start_segment: LineString,
        obstacle_segment: LineString,
        target_segment: LineString,
        current_point: Point | None,
        turning_radius: float,
        bounds: LinearRing,
        debug_prints: bool = False) -> tuple[LineString | None, LineString | None]:
    """Calculate tear around maneuver for a given segment.

    Connects start_segment → obstacle_segment → target_segment using Dubins curves.
    Falls back to direct concatenation if Dubins connection fails.

    Returns
    -------
        Tuple of (full_path, smoothed_evasion) where smoothed_evasion is the
        Dubins-smoothed arc portion only (gen_path_1 + obstacle_segment + gen_path_2).
    """
    try:
        gen_path_1 = None
        gen_path_2 = None
        gen_path_1, _, _ = geom_tools.get_curve(
            current_line_oriented=start_segment,
            target_line=obstacle_segment,
            turning_radius=turning_radius,
            step_size=32.0,
            robot_point=None,
            border_buffer=bounds,
            extend_start_line=True,
            extend_target_line=False)

        gen_path_2, _, _ = geom_tools.get_curve(
            obstacle_segment,
            target_segment,
            turning_radius,
            32.0,
            None,
            bounds,
            extend_start_line=True,
            extend_target_line=False)
        if gen_path_1 is None or gen_path_2 is None:
            return None, None
        if debug_prints:
            print(f"assemble_path_around_obstacle: gen_path_1={gen_path_1 is not None}, gen_path_2={gen_path_2 is not None}")

        if gen_path_1 is not None and gen_path_2 is not None:
            smoothed_evasion = geom_tools.combine_paths(current_path=gen_path_1, next_paths=[obstacle_segment, gen_path_2])
            gen_path = geom_tools.combine_paths(start_segment, [gen_path_1, obstacle_segment, gen_path_2, target_segment])
            return gen_path, smoothed_evasion

        # Fallback: direct concatenation when Dubins connection fails
        # The obstacle_segment endpoints already lie on start/target segments,
        # so a direct connection is geometrically valid.
        if debug_prints:
            print("assemble_path_around_obstacle: Dubins failed, using direct concatenation fallback")
        if gen_path_1 is not None:
            gen_path = geom_tools.combine_paths(start_segment, [gen_path_1, obstacle_segment])
            gen_path = (
                geom_tools.combine_paths(gen_path, [gen_path_2, target_segment]) if gen_path_2 is not None
                else gen_path
            )
        elif gen_path_2 is not None:
            gen_path = start_segment
            if gen_path is not None:
                obstacle_with_target = geom_tools.combine_paths(obstacle_segment, [gen_path_2, target_segment])
                if obstacle_with_target is not None:
                    gen_path = LineString([*gen_path.coords, *obstacle_with_target.coords])
        else:
            # Both Dubins failed — concatenate all segments directly
            gen_path = LineString([*start_segment.coords, *obstacle_segment.coords, *target_segment.coords])
        return gen_path, obstacle_segment

    except Exception as e:
        print(f"assemble_path_around_obstacle failed: {e}")
        return None, None


def insert_hooks_around_obs(path_around_obs: LineString,
                            obstacle: LinearRing,
                            inner_border: MultiPolygon,
                            outer_turning_head: LinearRing,
                            ab_line: LineString,
                            route_params: RoutePlanningConfig,
                            field_border: Polygon | LinearRing) -> LineString:
    """Calculates the working corridor error and insert the hook curves to the obstacle."""

    straight_buffer = route_params.min_straight_stop_distance_to_obstacle
    # make a Polygon without holes out of the LinearRing
    obstacle_exterior = obstacle.buffer(straight_buffer, quad_segs=32, cap_style='flat').exterior
    turning_headland: Polygon = Polygon(obstacle_exterior)  # poly without holes
    # this is for checking collisions at the side
    field_border_poly = (field_border.difference(Polygon(obstacle)) if isinstance(field_border, Polygon)
                         else Polygon(field_border, [obstacle]))
    # here I am missing support for multiple cutouts near each other
    turning_head_field = Polygon(outer_turning_head, [obstacle_exterior])
    inner_border_obs = inner_border.difference(turning_headland, grid_size=0)
    curve_start = Point(path_around_obs.coords[0])
    curve_end = Point(path_around_obs.coords[-1])
    corridor_error_start = curve_start.distance(obstacle)
    corridor_error_end = curve_end.distance(obstacle)
    # this assumes a circle buffer around the true obstacle
    oriented_ab = ab_line
    if path_around_obs.project(Point(ab_line.coords[0])) > path_around_obs.project(Point(ab_line.coords[-1])):
        oriented_ab = ab_line.reverse()

    def split_ab_at_obs(oriented_ab: LineString, obstacle: Polygon) -> Tuple[LineString | None, LineString | None]:
        """Splits the ab line into a before and after part at the obstalce."""
        split_ab = oriented_ab.difference(obstacle)
        if isinstance(split_ab, MultiLineString):
            split_ab = list(split_ab.geoms)
        if isinstance(split_ab, list):
            if len(split_ab) > 2:
                raise RoutePlanningError("Unexpected number of splitted ab lines.")
            ab_lines = sorted(split_ab, key= lambda ab: ab.distance(path_around_obs.boundary.geoms[0]))
            ab_line_before = ab_lines[0]
            ab_line_after = ab_lines[1]
        else:
            path_dist_ab = path_around_obs.project(split_ab.boundary.geoms[0])
            path_dist_obs = path_around_obs.project(obstacle.centroid)
            if path_dist_ab < path_dist_obs:
                ab_line_before = split_ab
                ab_line_after = None
            else:
                ab_line_before = None
                ab_line_after = split_ab
        return ab_line_before, ab_line_after

    ab_line_before, ab_line_after = split_ab_at_obs(oriented_ab, turning_headland)
    if ab_line_after is None or ab_line_before is None:
        ab_line_before, ab_line_after = split_ab_at_obs(
            oriented_ab, turning_headland. buffer(route_params.working_width / 2, quad_segs=32, cap_style='flat'))
    # I need to split the path in its curve part and its straight line parts
    path_start, curve, path_end = geom_tools.split_straight_endings(path_around_obs, ab_line)
    path_with_hooks = curve
    if path_with_hooks is None:
        return None
    # its a bit tricky here, regarding the correct calculation of the working corridor and the extensions
    if ab_line_before is not None:
        start_corridor = get_corridor_line(ab_line_before,
                                           route_params,
                                           obstacle_exterior,
                                           inner_border_obs,
                                           outer_turning_headland = outer_turning_head,
                                           field_border=field_border_poly)
        if (
            corridor_error_start > route_params.corridor_threshold and inner_border.intersects(ab_line_before)
            and not turning_headland.contains(ab_line_before)
                ):
            path_with_hooks = insert_hook_stops_to_headland(
                path_with_hooks, start_corridor, turning_head_field, route_params, field_border_poly)
    if ab_line_after is not None:
        end_corridor = get_corridor_line(ab_line=ab_line_after,
                                         route_params=route_params,
                                         turning_headland=obstacle_exterior,
                                         inner_border=inner_border_obs,
                                         outer_turning_headland=outer_turning_head,
                                         field_border=field_border_poly)
        if (
            corridor_error_end > route_params.corridor_threshold and inner_border.intersects(ab_line_after)
            and not turning_headland.contains(ab_line_after)
                ):
            # end_corridor = end_corridor.reverse()
            path_with_hooks = insert_hook_stops_to_ab(
                path_with_hooks, end_corridor, turning_head_field, route_params, field_border_poly)
    path_with_hooks = LineString(list(path_start.coords) + list(path_with_hooks.coords) + list(path_end.coords))
    return path_with_hooks


def determine_bounding_abs(
        obstacle: LinearRing | Polygon,
        ab_lines: MultiLineString,
        working_width: float,
        track_width: float) -> MultiLineString | None:
    """Determines the first two ab_lines that are not affected by the obstacle."""
    if working_width <= track_width:
        return None
    affected_ab_line_index = [
        index for index, ab_line in enumerate(ab_lines.geoms) if ab_line.distance(obstacle) < working_width / 2]
    if not affected_ab_line_index:
        return None
    first_index = min(affected_ab_line_index)
    last_index = max(affected_ab_line_index)
    bounding_index = [
        index for index in [first_index - 1 if first_index > 0 else None,
                            last_index + 1 if last_index < len(ab_lines.geoms) - 1 else None] if index is not None
                            ]
    if len(bounding_index) == 1:
        return MultiLineString([ab_lines.geoms[bounding_index[0]]])
    if len(bounding_index) == 2:
        return MultiLineString([ab_lines.geoms[index] for index in bounding_index])
    return None


def extend_start_end(route: Route) -> None:
    """
    Extends the route path at the start and end by up to 5m.

    Only applies per endpoint if that point is roughly inside the inner_border.
    The start extension is capped at the distance to the nearest headland.
    The end extension is capped at the distance to the farther AB line endpoint.
    """
    if route._path is None or len(route._path) < 2:
        return

    max_extension = 5.0
    inner_border = route._inner_border
    if inner_border is None:
        return

    def _is_inside(point: Point) -> bool:
        if isinstance(inner_border, MultiPolygon):
            return any(poly.buffer(0.1).contains(point) for poly in inner_border.geoms)
        elif isinstance(inner_border, Polygon):
            return inner_border.buffer(0.1).contains(point)
        else:
            return Polygon(inner_border).buffer(0.1).contains(point)

    # --- Extend at start ---
    start_point = route._path[0]
    if _is_inside(start_point):
        all_headland_rings = [ring for ring_list in route._track_system.headlands for ring in ring_list]
        nearest_headland_dist = min(
            ring.distance(start_point) for ring in all_headland_rings
        )
        extension_distance = min(max_extension, nearest_headland_dist)
        if extension_distance > 0.01:
            first_segment = LineString([route._path[1], route._path[0]])
            direction = geom_tools.direction_of_line(first_segment)
            new_start = Point(
                start_point.x + direction[0] * extension_distance,
                start_point.y + direction[1] * extension_distance
            )
            route._path.insert(0, new_start)

    # --- Extend at end ---
    end_point = route._path[-1]
    if _is_inside(end_point):
        closest_node = min(route._nodes, key=lambda node: node.point.distance(end_point))
        closest_ab_line = closest_node.ab_line

        ab_start = Point(closest_ab_line.coords[0])
        ab_end = Point(closest_ab_line.coords[-1])
        farther_endpoint = max(ab_start, ab_end, key=lambda p: p.distance(end_point))

        extension_distance = min(max_extension, end_point.distance(farther_endpoint))
        if extension_distance > 0.01:
            last_segment = LineString([route._path[-2], route._path[-1]])
            direction = geom_tools.direction_of_line(last_segment)
            new_end = Point(
                end_point.x + direction[0] * extension_distance,
                end_point.y + direction[1] * extension_distance
            )
            route._path.append(new_end)

    _rebuild_line(route, route._path)


def interpolate_ab_lines(route: Route) -> None:
    """
    Fills large distances in the path with more points.
    """
    if route._line is None:
        return
    if len(route._line.coords) < 2:
        return
    interval = 25.0

    line: LineString = route._line
    route._line = line.segmentize(interval)
    route._path = [Point(c) for c in route._line.coords]


def simplify_points(route: Route) -> None:
    """
    Deletes double points within a margin from path.
    """
    minimum_distance = 0.01
    if route._path is None or len(route._path) == 0:
        return
    new_points = []

    i = 0
    while i < len(route._path) - 1:
        p1 = route._path[i]
        ii = i + 1
        p2 = route._path[ii]
        while ii < len(route._path) - 1 and p1.distance(p2) < minimum_distance:
            ii += 1
            p2 = route._path[ii]

        new_points.append(p1)
        new_points.append(p2)
        i = ii + 1

    if (new_points[0].distance(route._path[-1]) > minimum_distance
            and new_points[-1].distance(route._path[-1]) > minimum_distance):
        new_points.append(route._path[-1])
    _rebuild_line(route, new_points)


def _rebuild_line(route: Route, new_points):
    """
    Helper function to rebuild the routes linestring from an array of points.
    """
    route._path = new_points
    coords = []
    coords.extend(new_points)
    if route._route_params.round_trip_route:
        coords.append(new_points[0])
    route._line = LineString(coords)


def collision_in_route(route: Route) -> bool:
    """
    Checks the nearest distance of the path to the boundaries of the field.
    """

    track = route._line

    if track is None:
        return False

    border = route._track_system.outer_border
    collider = MultiLineString([border.exterior] + list(border.interiors))
    distance_margin = (route._route_params.working_width / 2) - 0.001

    if track.dwithin(collider, distance_margin):
        if route._route_params.debug_prints:
            print("Path has potential collisions")
        return True

    if route._route_params.debug_prints:
        print("Path is clear of potential collisions")

    return False


FUNCTIONS = {
    PostSteps.CUTOUT_AVOIDANCE: cutout_avoidance,
    PostSteps.EVADE_OBSTACLES: evade_rois,
    PostSteps.EXTEND_START_END: extend_start_end,
    PostSteps.AB_LINE_INTERPOLATION: interpolate_ab_lines
}
