
from dataclasses import dataclass
from typing import List

from shapely import LinearRing, LineString, MultiLineString, MultiPoint, Point
from shapely.ops import substring
from shapely import remove_repeated_points

from nexat_trace.planning import curve_calculation
from nexat_trace.planning.route import Route
from nexat_trace.shared.config import PostSteps
from nexat_trace.util import geom_tools

"""
This module defines post processing steps for routes

Defined functions must take only the Route as a parameter and
act in place.
The RoutePlanner automatically looks up all available steps
and their corresponding functions in the FUNCTIONS dictionary.
"""


def cutout_avoidance(route: Route) -> None:
    """
    Checks a given Route if the path intersects with any specialized cutout segments and changes the path accordingly.
    """

    @dataclass
    class Intersect:
        start: Point
        end: Point
        segment: LinearRing

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
    for segment in cutout_segments:

        intersections = original_path.intersection(segment)

        # Intersections
        if isinstance(intersections, (MultiPoint, LineString)):
            if intersections.is_empty:
                continue

            if isinstance(intersections, LineString):
                intersections = [
                    Point(intersections.coords[0]),
                    Point(intersections.coords[-1]),
                ]

            intersections = list(intersections.geoms)
            intersections.sort(key=lambda intersect: original_path.project(intersect))

            # Intersection at start / end
            if len(intersections) % 2 != 0:

                if (original_path.project(Point(intersections[1]))
                        - original_path.project(Point(intersections[0]))
                        > original_path.project(Point(intersections[-1]))
                        - original_path.project(Point(intersections[-2]))):

                    start_end.append(intersections.pop(0))

                else:
                    start_end.append(intersections.pop(-1))

            # Intersection at headland
            if (len(intersections) > 4 and
                    original_path.project(Point(intersections[-1]))
                    - original_path.project(Point(intersections[0]))
                    < 4 * route._route_params.working_width  # Circle circumference approximation
                    and route._field_border.dwithin(segment, route._route_params.working_width / 2)):
                all_intersections.append(Intersect(intersections[0], intersections[-1], segment))
                continue

        # Start / Endpoint ?
        elif isinstance(intersections, Point):
            start_end.append(intersections)

        for i in range(0, len(intersections), 2):
            all_intersections.append(Intersect(intersections[i], intersections[i + 1], segment))

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

            segment_path_points = (
                [Point(coord) for coord in curve_on_ab_segment.coords]
                + [Point(coord) for coord in curve_on.coords]
                + cut_segment_path_points
                + [Point(coord) for coord in curve_off.coords]
                + [Point(coord) for coord in rest_path.coords]
            )

            final_path.extend(segment_path_points)
            if len(geom_tools.segment_line(LineString(segment_path_points), radius_threshold = 13.5 / 10.0)[0]) > 1:
                print("Segmented line detected after cutout avoidance, consider increasing segmentation_avoidance_distance or decreasing vehicle_turning_radius")
            last_intersection = intersection

    _rebuild_line(route, final_path)


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
    PostSteps.AB_LINE_INTERPOLATION: interpolate_ab_lines
}
