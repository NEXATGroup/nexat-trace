from itertools import tee
from math import pi
from typing import Iterable, Iterator, List, Tuple, TypeVar

import dubins
import numpy as np
from numpy import typing as npt
from shapely import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    remove_repeated_points,
    simplify,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, substring, unary_union


class Circle:
    """
    Class for circle geometries.
    """

    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = center
        self.radius = radius

    def get_polygon(self, resolution = 40) -> Polygon:
        """
        Returns the circle in a polygon representation.
        """
        return Point(self.center).buffer(self.radius, resolution = resolution)


def direction_of_line(line: LineString) -> npt.NDArray:
    """
    Returns a normalized vector of the direction of the first point of a line to the last.
    """
    p1 = np.array(line.coords[0])
    p2 = np.array(line.coords[-1])
    vec = p2 - p1
    return vec / np.linalg.norm(vec)


def angle_of_line(line: LineString) -> np.floating:
    """
    Returns the angle of a line.
    """
    a = Point(line.coords[1])
    b = Point(line.coords[0])
    return np.angle((a.x - b.x) + (a.y * 1j - b.y * 1j))


def angle_between_angles(angle1: float, angle2: float) -> float:
    """
    Returns the angle between two angles.
    """
    result_phi = angle2 - angle1
    y = np.sin(result_phi)
    x = np.cos(result_phi)
    return np.arctan2(y, x)


def angle_between_lines(line: LineString, line2: LineString) -> float:
    """
    Returns the angle between two lines.
    """
    return angle_between_angles(angle_of_line(line), angle_of_line(line2))


def extend_line_in_bounds(
        line: LineString,
        bounds: Polygon | LinearRing,
        max_distance: float | None = None,
        extend_front: bool = True,
        extend_back: bool = True) -> LineString:
    """
    Returns a version of the line with back and front extensions touching the the first intersection with the bounds.
    """
    line = remove_repeated_points(line, 1e-4)
    bounds_exterior_ring = bounds
    bounds_collider = bounds
    if isinstance(bounds, Polygon):
        bounds_exterior_ring = bounds.exterior
        bounds_collider = MultiLineString([bounds.exterior] + list(bounds.interiors))

    # get outwards pointing end segments of line
    start_segment = LineString([line.coords[1], line.coords[0]])
    end_segment = LineString([line.coords[-2], line.coords[-1]])

    distance = bounds_exterior_ring.length / 2.0
    if max_distance is not None:
        distance = max_distance

    # extend segments & get intersection points with bounds
    back_extension = LineString(
        [
            start_segment.coords[-1],
            start_segment.coords[-1] + direction_of_line(start_segment) * distance
        ]
    )

    front_extension = LineString(
        [
            end_segment.coords[-1],
            end_segment.coords[-1] + direction_of_line(end_segment) * distance
        ]
    )

    back_intersection = back_extension.intersection(bounds_collider)
    back_intersection_point: Point | None = None

    if isinstance(back_intersection, Point):
        back_intersection_point = back_intersection

    elif isinstance(back_intersection, (MultiPoint, LineString)):
        points = []
        if isinstance(back_intersection, MultiPoint):
            points = list(back_intersection.geoms)

        elif not back_intersection.is_empty:
            points = [Point(c) for c in back_intersection.coords]

        elif back_intersection.is_empty:
            points = [Point(back_extension.coords[-1])]

        back_intersection_point = min(points, key = lambda p: back_extension.project(p))

    front_intersection = front_extension.intersection(bounds_collider)
    front_intersection_point: Point | None = None

    if isinstance(front_intersection, Point):
        front_intersection_point = front_intersection

    elif isinstance(front_intersection, (MultiPoint, LineString)):
        points = []
        if isinstance(front_intersection, MultiPoint):
            points = list(front_intersection.geoms)

        elif not front_intersection.is_empty:
            points = [Point(c) for c in front_intersection.coords]

        elif front_intersection.is_empty:
            points = [Point(front_extension.coords[-1])]

        front_intersection_point = min(points, key = lambda p: front_extension.project(p))

    new_coords = [Point(c) for c in line.coords]
    if back_intersection_point is not None and extend_back:
        new_coords.insert(0, back_intersection_point)

    if front_intersection_point is not None and extend_front:
        new_coords.append(front_intersection_point)

    return LineString(new_coords)


def extend_line(line: LineString, distance: float, extend_front: bool = True, extend_back: bool = True) -> LineString:
    """
    Extends the given line.
    """
    line = remove_repeated_points(line, 1e-4)
    # get outwards pointing end segments of line
    start_segment = LineString([line.coords[1], line.coords[0]])
    end_segment = LineString([line.coords[-2], line.coords[-1]])

    coords = []

    # extend segments & get intersection points with bounds
    if extend_back:
        coords.append(start_segment.coords[-1] + direction_of_line(start_segment) * distance)

    coords.extend(list(line.coords))

    if extend_front:
        coords.append(end_segment.coords[-1] + direction_of_line(end_segment) * distance)

    return LineString(coords)


def ring_with_origin_at(ring: LinearRing, new_origin_point: Point):
    """
    Returns a new LinearRing instance with its origin at a given point on the ring.
    """
    part1 = substring(
        ring,
        ring.project(new_origin_point, True),
        1.0,
        True
    )
    part2 = substring(
        ring,
        0.0,
        ring.project(new_origin_point, True),
        True
    )
    coords = list(part1.coords) + list(part2.coords)[1:-1]
    new_ring = LinearRing(coords)
    return new_ring


def get_substring_on_linearring(ring, start_point, stop_point) -> LineString:
    """
    Returns a substring of the ring safe against errors due to ring wrapping.
    """
    aligned_ring = ring_with_origin_at(ring, start_point)
    return substring(
        aligned_ring,
        0.0,
        aligned_ring.project(stop_point)
    )


def erode_linearring(ring: LinearRing, radius: float) -> LinearRing:
    """
    Smooths the linear ring to be drivable for the vehicle.

    Erodes the given linearring by buffering outwards once, inwards twice and outwards once again
    to smooth all vertices in the geometry to the given radius.

    """
    buffer = Polygon(
        ring.buffer(
            1.0 * radius,
            join_style="round",
            resolution=45
        ).exterior
    )
    buffer = buffer.buffer(
        -2.0 * radius,
        join_style="round",
        resolution=45
    )

    if isinstance(buffer, MultiPolygon):
        buffer = max(buffer.geoms, key=lambda poly: poly.area)

    if not isinstance(buffer, Polygon):
        raise ValueError("Failed to smooth headland ring in one piece")

    new_ring = buffer.buffer(
        1.0 * radius,
        join_style="round",
        resolution=45
    ).exterior

    return new_ring


def erode_polygon_inwards(poly: Polygon, radius: float) -> Polygon:
    """
    Smooths the Polygon to be drivable for the vehicle.

    Erodes the given polygon by buffering inwards once, outwards once
    to smooth all vertices in the geometry to the given radius.

    """
    buffer = poly.buffer(
        -1.0 * radius,
        join_style="round",
        resolution=45
    )
    buffer = buffer.buffer(
        +1.0 * radius,
        join_style="round",
        resolution=45
    )

    if isinstance(buffer, MultiPolygon):  # TODO handle this case better
        buffer = max(buffer.geoms, key=lambda poly: poly.area)

    if not isinstance(buffer, Polygon):
        raise ValueError("Failed to smooth headland ring in one piece")

    return buffer


T = TypeVar('T')


def triplewise(iterable: Iterable[T]) -> Iterator[tuple[T, T, T]]:
    """
    Get overlapping 3-tuples of the items in the `input` iterable.

    The number of items in the output iterator is two fewer than in the input `iterable`.
    Thus, `iterable` should contain at least three items in order for the output to not be empty.

    Parameters
    ----------

    iterable : Iterable
        Target iterable

    Returns
    -------

        Iterator over triples of items from the input iterable

    Example
    -------

    ```python
    for first, second in pairwise(range(5)):
        print(f"({first}, {second}, {third})")
    # Prints:
    # (0, 1, 2)
    # (1, 2, 3)
    # (2, 3, 4)
    ```
    """
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c, strict = False)


def calculate_circle(points: List[Tuple[float, float]]) -> Circle:
    """
    Calculates a circle that is fitted to three points.
    """
    try:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
        radius = np.sqrt((ux - x1) ** 2 + (uy - y1) ** 2)
        return Circle((ux, uy), radius)
    except ZeroDivisionError:
        return Circle((0, 0), float("inf"))


def line_discontinuities(
        line: LineString,
        angle_threshold: float = pi / 6.0,
        radius_threshold: float = 12.0) -> list[Point]:
    """
    Get the locations of discontinuities in a line. Expects a line without duplicates.

    Parameters
    ----------

    line : LineString
        Line to check for discontinuities.

    angle_threshold : float
        The maximum angle between two consecutive segments of the line.

    radius_threshold : float
        The maximum radius of the circle that can be fitted to three consecutive points.

    Returns
    -------

        List of points with the locations of discontinuities
    """
    if line is None or line.is_empty or not isinstance(line, LineString):
        return []

    line_lines = triplewise(list(line.coords))
    segmentation_points = []

    for i, points in enumerate(line_lines):
        # Skip degenerate triples where consecutive points are identical
        if points[0] == points[1] or points[1] == points[2]:
            continue

        circle = calculate_circle([*points])
        try:
            angle = abs(angle_between_lines(
                LineString([points[0], points[1]]),
                LineString([points[1], points[2]])
            ))
        except Exception:
            print(f"Error calculating angle between lines for points: {points}")
            continue
        if circle.radius < radius_threshold or angle > angle_threshold:
            segmentation_points.append((i, Point(points[1])))

    return segmentation_points


def segment_line(
        input_linestring: LineString,
        angle_threshold: float = pi / 6.0,
        radius_threshold: float = 12.0,
        recurse_once: bool = False) -> tuple[list, MultiPoint]:
    """
    Cut a given line into segments. Removes duplicate points.

    Parameters
    ----------

    input_linestring : LineString
        Line to segment.

    angle_threshold : float
        The maximum angle between two consecutive segments of the line.

    radius_threshold : float
        The maximum radius of the circle that can be fitted to three consecutive points.

    recurse_once : bool
        If True, the function will recurse once if no segmentation points are found.

    Returns
    -------

        A tuple with a list of segmented lines and a MultiPoint with the points of discontinuities.
    """
    if input_linestring is None or input_linestring.is_empty or not isinstance(input_linestring, LineString):
        return [], []
    free_of_duplicates: LineString | None = None
    try:
        free_of_duplicates = remove_repeated_points(input_linestring, 0.01)
        if not free_of_duplicates.is_valid:
            print("Geometry is not valid after removing duplicates.")
            # free_of_duplicates = input_linestring
    except Exception:
        free_of_duplicates = input_linestring
    if len(free_of_duplicates.coords) < 2:
        return [], []
    segmentation_points = line_discontinuities(free_of_duplicates, angle_threshold, radius_threshold)

    if not segmentation_points:
        return [free_of_duplicates], []

    segments: list[LineString] = []
    last_elem: int = 0

    for elem_num, _ in segmentation_points:
        coords = list(free_of_duplicates.coords[last_elem:elem_num + 2])
        if len(coords) >= 2:
            segment = LineString(coords)
            if segment.length > radius_threshold:
                segments.append(segment)
        last_elem = elem_num + 1

    # Handle remaining tail
    tail_coords = list(free_of_duplicates.coords[last_elem:])
    if len(tail_coords) >= 2:
        leftovers = LineString(tail_coords)
        if leftovers.length > radius_threshold:
            if recurse_once:
                leading = segments.pop(0) if segments else None
                combined_coords = tail_coords[:]
                if leading is not None:
                    combined_coords.extend(list(leading.coords))
                if len(combined_coords) >= 2:
                    rest_segments, rest_points = segment_line(LineString(combined_coords))
                    segments.extend(rest_segments)
                    segmentation_points.extend(rest_points)
            else:
                segments.append(leftovers)

    if not segments:
        return [free_of_duplicates], []

    return segments, segmentation_points


def line_buffer_intersection(
        line1: LineString,
        line2: LineString,
        radius: float) -> MultiPoint | None:
    """
    Returns intersection points between line buffers.
    """
    # extend lines until they intersect
    line1_buffer = line1.buffer(radius, cap_style=1, join_style=1).boundary
    line2_buffer = line2.buffer(radius, cap_style=1, join_style=1).boundary
    intersection: MultiPoint = line1_buffer.intersection(line2_buffer)

    if intersection.is_empty:
        return None

    if isinstance(intersection, MultiPoint):
        return intersection

    return None


def dubins_between_vectors(
    in_vector: LineString, out_vector: LineString, curve_radius: float
) -> LineString:
    """
    Return the length of the dubins path in between two points.
    """
    q1 = (
        in_vector.coords[1][0],
        in_vector.coords[1][1],
        (angle_of_line(in_vector))
    )
    q2 = (
        out_vector.coords[0][0],
        out_vector.coords[0][1],
        (angle_of_line(out_vector)),
    )
    path, _ = dubins.shortest_path(q1, q2, curve_radius).sample_many(0.5)
    point_list = []
    for pose in path:
        point_list.append(Point(pose[0], pose[1]))
    point_list.append(Point(q2[0], q2[1]))
    return LineString(point_list)


def dubins_between_segments(
        current_line: LineString,
        target_line: LineString,
        turning_radius: float,
        current_point: Point = None,
        bounds: LinearRing = None,
        headland_ring: LinearRing = None,
        ab_is_current_line: bool = False,
        extend_start_line: bool = True,
        exhaustive: bool = False) -> LineString | None:
    """
    Return a dubins curve between two line segments.
    """

    if current_line is None:
        return None
    if target_line is None:
        return None
    if turning_radius is None or turning_radius <= 0.0:
        return None

    turning_radius = abs(turning_radius)

    current_line_extended = current_line
    if current_line.coords[0] != current_line.coords[-1] and extend_start_line:
        current_line_extended = extend_line_in_bounds(current_line_extended, bounds, extend_back = False)

    target_line_extended = target_line
    cnt = 0
    while current_line.intersects(target_line_extended) and cnt < 20:
        target_line_extended = substring(target_line_extended, 0.05, target_line_extended.length)
        cnt += 1

    if target_line.coords[0] != target_line.coords[-1]:
        target_front_extended = extend_line_in_bounds(target_line_extended, bounds, extend_front=True, extend_back=False)
        cnt = 0
        while current_line.intersects(target_front_extended) and cnt < 20:
            target_front_extended = substring(target_front_extended, 0, target_front_extended.length * 0.95)
            cnt += 1

        target_line_extended = target_front_extended
        target_line_extended = extend_line_in_bounds(target_line_extended, bounds, extend_front=False, extend_back=True)

    intersection_point_line = current_line_extended.intersection(target_line_extended)
    current_point_projection = current_line_extended.project(current_point)
    nearest_intersection_point_to_point = None
    intersection_point_line_list = []
    if isinstance(intersection_point_line, Point):
        nearest_intersection_point_to_point = intersection_point_line
        # here we only need this for the while loop
        intersection_point_line_list.append(
            (
                current_line_extended.project(nearest_intersection_point_to_point) - current_point_projection,
                nearest_intersection_point_to_point,
            )
        )

    elif isinstance(intersection_point_line, MultiPoint):
        for point in intersection_point_line.geoms:
            if current_line_extended.project(point) - current_point_projection > 0:
                intersection_point_line_list.append(
                    (
                        current_line_extended.project(point) - current_point_projection,
                        point,
                    )
                )
        if len(intersection_point_line_list) == 0:
            return None
        intersection_point_line_list.sort(key=lambda x: x[0])
        nearest_intersection_point_to_point = intersection_point_line_list[0][1]

    intersection_points_buffer: MultiPoint = line_buffer_intersection(
        current_line_extended, target_line_extended, turning_radius + 0.1
    )
    if intersection_points_buffer is None:
        return None

    ipol_candidates = []
    # TODO find a better check. This might have problems with very short ab's and the outermost headland ring.
    if headland_ring is not None:
        poly_head_ring = Polygon(headland_ring)
        line_to_check = current_line if ab_is_current_line else target_line
        intersection = poly_head_ring.intersection(line_to_check)
        is_current_line_contained = (
            intersection.length / line_to_check.length > 0.3
            if line_to_check.length > 0
            else poly_head_ring.contains(line_to_check)
        )
    cnt = 0
    while cnt < len(intersection_point_line_list) and len(ipol_candidates) == 0:
        for intersection_point in intersection_points_buffer.geoms:
            ipol_current = current_line_extended.interpolate(current_line_extended.project(intersection_point))
            ipol_target = target_line_extended.interpolate(target_line_extended.project(intersection_point))

            if intersection_point_line.is_empty or (
                current_line_extended.project(ipol_current) < current_line_extended.project(nearest_intersection_point_to_point)
                and target_line_extended.project(ipol_target) > target_line_extended.project(nearest_intersection_point_to_point)
                and current_line_extended.project(ipol_current) - current_point_projection > 0
                and (headland_ring is None or poly_head_ring.contains(intersection_point) == is_current_line_contained)
            ):
                ipol_candidates.append(
                    (
                        current_line_extended.project(ipol_current) - current_point_projection,
                        intersection_point,
                        ipol_current,
                        ipol_target,
                    )
                )
        if not exhaustive:
            break
        cnt += 1
        if cnt < len(intersection_point_line_list):
            nearest_intersection_point_to_point = intersection_point_line_list[cnt][1]
    if ipol_candidates == []:
        return None
    ipol_candidates.sort(key=lambda x: x[0])
    for _, _, ipol_current, ipol_target in ipol_candidates:
        current_segment_new_coords = substring(
            current_line_extended,
            0.0,
            current_line_extended.project(ipol_current)
        ).coords[-2:]

        if len(current_segment_new_coords) < 2:
            continue
        try:
            current_segment = LineString(current_segment_new_coords)

            target_segment = LineString(
                substring(
                    target_line_extended,
                    target_line_extended.project(ipol_target),
                    target_line_extended.length
                ).coords[:2]
            )
        except Exception:
            continue
        current_vector = LineString((current_segment.boundary.geoms[0], ipol_current))
        target_vector = LineString((ipol_target, target_segment.boundary.geoms[-1]))
        if abs(angle_between_lines(current_vector, current_line)) > pi:
            continue
        if abs(angle_between_lines(target_vector, target_line)) > pi:
            continue

        candidate_turning_radius = turning_radius
        if abs(angle_between_lines(current_vector, target_vector)) < pi / 4:
            candidate_turning_radius -= 0.5
        path = dubins_between_vectors(current_vector, target_vector, candidate_turning_radius)
        if path is not None and path.length > turning_radius * pi:
            continue
        return path

    return None


def turn_to_segment(
        from_segment: LineString,
        to_segment: LineString,
        outer_border: LinearRing,
        turning_radius: float,
        extend_to_segment_front: bool) -> LineString | None:
    """
    Generate a turn to target segment.

    Simpler option to dubins_between_segments.
    """

    line1 = from_segment
    line2 = to_segment

    if not line1.intersects(line2):
        # extend lines to try to get intersection
        line1 = extend_line_in_bounds(line1, outer_border, extend_back = False)
        line2 = extend_line_in_bounds(
            line2,
            outer_border,
            extend_front = extend_to_segment_front,
        )

        if not line1.intersects(line2):
            return None

    # get curve pivot location
    line1_offset_r = line1.parallel_offset(turning_radius)
    line1_offset_l = line1.parallel_offset(-turning_radius)
    line1_offset = line1_offset_r
    line2_end = to_segment.boundary.geoms[-1]
    if line1_offset_l.centroid.distance(line2_end) < line1_offset_r.centroid.distance(line2_end):
        line1_offset = line1_offset_l

    line1_start = from_segment.boundary.geoms[0]
    line2_offset_r = line2.parallel_offset(turning_radius)
    line2_offset_l = line2.parallel_offset(-turning_radius)
    line2_offset = line2_offset_r
    if line2_offset_l.centroid.distance(line1_start) < line2_offset_r.centroid.distance(line1_start):
        line2_offset = line2_offset_l

    pivot_location: Point | None = None
    intersection = line1_offset.intersection(line2_offset)
    if isinstance(intersection, Point):
        pivot_candidates = [intersection]
    elif isinstance(intersection, MultiPoint):
        # Sort candidates to prefer pivots that maximize drive distance on line1
        pivot_candidates = sorted(intersection.geoms, key=lambda p: from_segment.project(p), reverse=True)
    else:
        return None
    for pivot_location in pivot_candidates:
        turn_circle: LinearRing = pivot_location.buffer(turning_radius, resolution = 45).exterior
        nearest_point_line1, _ = nearest_points(line1, turn_circle)
        nearest_point_line2, _ = nearest_points(line2, turn_circle)
        turn_circle = ring_with_origin_at(turn_circle, nearest_point_line1)
        nearest_point_line1_projection = line1.project(nearest_point_line1)
        line1_direction = LineString(
            [
                line1.interpolate(nearest_point_line1_projection - 0.01),
                line1.interpolate(nearest_point_line1_projection + 0.01)
            ]
        )
        turn_circle_direction = LineString(
            [
                turn_circle.coords[0],
                turn_circle.coords[1],
            ]
        )
        if abs(angle_between_lines(line1_direction, turn_circle_direction)) > pi / 2.0:
            turn_circle = turn_circle.reverse()

        curve = get_substring_on_linearring(turn_circle, nearest_point_line1, nearest_point_line2)

        if curve.length > 1.8 * pi * turning_radius:
            continue

        curve_start = LineString(
            [
                curve.coords[0],
                curve.coords[1]
            ]
        )
        if abs(angle_between_lines(curve_start, line1)) > pi / 2:
            continue

        curve_end = LineString(
            [
                curve.coords[-2],
                curve.coords[-1]
            ]
        )
        if abs(angle_between_lines(curve_end, line2)) > pi / 2:
            continue

        return curve

    return None


def get_tangent_at_nearest_point(p: Point, line: LineString | LinearRing) -> LineString:
    """
    Returns tangent of a line at some point.
    """
    closest_point = line.interpolate(line.project(p))

    tangent_point_plus = line.interpolate(line.project(closest_point) + 0.01)
    tangent_point_minus = line.interpolate(line.project(closest_point) - 0.01)

    tangent = LineString([tangent_point_minus, tangent_point_plus])
    normalized = direction_of_line(tangent)
    return LineString(
        [
            closest_point,
            Point(
                [
                    closest_point.x + normalized[0],
                    closest_point.y + normalized[1]
                ]
            )
        ]
    )


def union_intersecting_geoms(geometries: List[BaseGeometry]) -> List[Polygon]:
    """
    Find groups of geometries that intersect with each other and returns them as unions.

    Elements of a group only need to intersect with one other element of the group.

    Parameters
    ----------

    geometries : List[BaseGeometry]
        List of geometries to be grouped.

    Returns
    -------

        List of grouped geometries.
    """
    groups = []
    intersections_found = False
    while geometries:
        base_geom = geometries.pop(0)
        group = [base_geom]
        non_intersecting = []
        for geom in geometries:
            if base_geom.intersects(geom):
                group.append(geom)
                intersections_found = True
            else:
                non_intersecting.append(geom)
        geometries = non_intersecting
        groups.append(group)
    grouped_geoms = [unary_union(group) for group in groups]

    if intersections_found:
        return union_intersecting_geoms(grouped_geoms)

    return grouped_geoms


def check_segmentation(path: List[Point] | LineString | List[LineString],
                       turning_headland: LinearRing | LineString,
                       function_string: str,
                       geom_from: BaseGeometry | None = None,
                       geom_to: BaseGeometry | None = None) -> bool:
    """Checks if the given path is segmeted correclty."""
    segments = []
    if isinstance(path, list) and (isinstance(path[0], Tuple) or path[0].geom_type == "Point"):
        path = LineString(path)
        segments, _ = segment_line(path, radius_threshold=1.35)
    elif isinstance(path, list) and path[0].geom_type == "LineString":
        segments = path
    elif isinstance(path, LineString):
        segments, _ = segment_line(path, radius_threshold=1.35)
    for i in range(len(segments) - 1):
        is_same_start_stop = segments[i].boundary.geoms[-1].distance(segments[i + 1].boundary.geoms[0]) < 1e-9
        is_direction_change = abs(
            angle_between_lines(LineString(segments[i].coords[-2:]), LineString(segments[i + 1].coords[:2]))
            ) > 0.95 * pi
        if not is_same_start_stop or not is_direction_change:
            path_start_dist = path.project(segments[i].boundary.geoms[-1])
            path_end_dist = path.project(segments[i + 1].boundary.geoms[0])
            path_between = substring(path, path_start_dist, path_end_dist)
            path_between = path_between  # to fix ruff errors
            print("Segmented path is not segmented correctly")
            return False
    return True


def recombine_ml_in_bounds(ml: MultiLineString,
                           origin: Point,
                           bound: LinearRing | Polygon) -> LineString:
    """Recombines a MultiLineString into a LineString that does not cross the bounds.

    The closest segment to the origin is guaranteed to be included in the resulting LineString.
    Assumes that the MultiLineString is in order.
    """
    if ml is None or ml.is_empty:
        return LineString()

    geoms = list(ml.geoms)
    if not geoms:
        return LineString()

    if len(geoms) == 1:
        return geoms[0]

    def _can_connect_without_crossing(connection_line: LineString, bound_poly: Polygon) -> bool:
        """Check if a connection line can be made without crossing the polygon boundary."""
        return not bound_poly.exterior.intersects(connection_line)

    bound_poly = bound if isinstance(bound, Polygon) else Polygon(bound)
    closest_idx = min(range(len(geoms)), key=lambda i: geoms[i].distance(origin))

    # Start building result with the closest segment
    result_coords = list(geoms[closest_idx].coords)

    # Try to extend forward from the closest segment
    current_idx = closest_idx
    while current_idx < len(geoms) - 1:
        next_idx = current_idx + 1

        next_segment = geoms[next_idx]
        end_point = Point(result_coords[-1])
        start_point = Point(next_segment.coords[0])

        connection = LineString([end_point, start_point])
        if _can_connect_without_crossing(connection, bound_poly):
            result_coords.extend(next_segment.coords[1:])
            current_idx = next_idx
        else:
            break

    # Try to extend backward from the closest segment
    current_idx = closest_idx - 1
    while current_idx > 0:

        prev_segment = geoms[current_idx]
        end_point = Point(prev_segment.coords[-1])
        start_point = Point(result_coords[0])

        # Check if connecting would cross the boundary
        connection = LineString([end_point, start_point])
        if _can_connect_without_crossing(connection, bound_poly):
            result_coords = list(prev_segment.coords[:-1]) + result_coords
            current_idx -= 1
        else:
            break

    return LineString(result_coords) if len(result_coords) >= 2 else LineString()


def direction_sgn_point_from_line(ref_line: LineString, target_point: Point) -> int:
    """Determines the sign of the direction, the point lies away from the reference line."""
    x1, y1 = ref_line.coords[0]
    x2, y2 = ref_line.coords[-1]
    cross = (x2 - x1) * (target_point.y - y1) - (y2 - y1) * (target_point.x - x1)
    sign = 1
    if cross < 0:
        sign = -1
    return sign


def combine_paths(current_path: LineString, next_paths: List[LineString], safety_distance: float | None = None):
    """Combine multiple LineStrings into one."""
    sd = safety_distance if safety_distance is not None else 0.1
    compound = current_path
    for next_path in next_paths:
        np_cu_pa, np_ne_pa = nearest_points(compound, next_path)
        end_dist = compound.project(np_cu_pa) - sd
        current_path_cut = substring(geom=compound, start_dist=0, end_dist=end_dist)
        start_dist = next_path.project(np_ne_pa) + sd
        next_path_cut = substring(geom=next_path, start_dist=start_dist, end_dist=next_path.length)
        compound = LineString([*current_path_cut.coords, *next_path_cut.coords])
    return compound


def get_curve(
        current_line_oriented: LineString,
        target_line: LineString,
        turning_radius: float,
        step_size: float,
        robot_point: Point | None,
        border_buffer: LinearRing | None,
        *,
        extend_start_line: bool = True,
        extend_target_line: bool = True) -> tuple[LineString, Point, list[Point]]:
    """Return a curve between two line segments.

    Args:
        current_line_oriented: The starting line segment
        target_line: The target line segment to connect to
        turning_radius: The turning radius for the curve
        step_size: Sampling step size for the path
        robot_point: Current machine position or None to use the whole lines
        border_buffer: Border for line extension limits or None
        extend_start_line: Whether to extend the start line
        extend_target_line: Whether to extend the target line

    Returns
    -------
        Tuple of (path, intersection_point, [ipol_current, ipol_target]) or (None, None, None) on failure
    """
    if not all({
        current_line_oriented and isinstance(current_line_oriented, LineString),
        target_line and isinstance(target_line, LineString),
        turning_radius and turning_radius > 0.0,
        step_size and step_size > 0.0,
        (robot_point is None or isinstance(robot_point, Point)),
        (border_buffer is None or isinstance(border_buffer, LinearRing))
            }):
        print("get_curve: Invalid input parameters")
        return None, None, None
    current_line = (
        current_line_oriented if robot_point is None
        else substring(current_line_oriented, current_line_oriented.project(robot_point), current_line_oriented.length)
    )
    if not isinstance(current_line, LineString):
        print("get_curve: robot_point projects to the end of current_line, resulting segment is not a LineString")
        return None, None, None
    line_distance = max(current_line.distance(target_line), turning_radius)

    if extend_start_line:
        current_line = extend_line(
            current_line, distance=line_distance * 2, extend_front=extend_start_line, extend_back=False
        )
        if isinstance(current_line, MultiLineString):
            current_line = LineString([coord for line in current_line.geoms for coord in line.coords])

    if extend_target_line:
        target_line = extend_line(
            target_line, distance=line_distance * 2, extend_front=extend_target_line, extend_back=extend_target_line
        )
        if isinstance(target_line, MultiLineString):
            target_line = LineString([coord for line in target_line.geoms for coord in line.coords])

    buffering_sign = relative_orientation(current_line, target_line, turning_radius)

    pivot_point = first_intersection(
        current_line=current_line.offset_curve(buffering_sign * turning_radius),
        intersector=target_line.offset_curve(buffering_sign * turning_radius)
        )

    arc = pivot_point.buffer(turning_radius, step_size).exterior
    arc_start = tangential_intersection(current_line, arc, turning_radius * 0.01)
    arc_end = tangential_intersection(target_line.reverse(), arc, turning_radius * 0.01)

    arc = ring_with_origin_at(arc, arc_start)

    if buffering_sign > 0:
        arc = arc.reverse()

    arc = substring(arc, 0, arc.project(arc_end))

    return arc, pivot_point, [arc_start, arc_end]


def tangential_intersection(current_line: LineString, circle: LinearRing, approximation_correction: float = 0.1):
    """Returns a point on the line, that is also on the circle.

    Args:
        current_line: The line that gets touched by the circle
        circle: The circle touching the line
        approximation_correction: The distance from the line in which the function looks for intersections with the circle

    Returns
    -------
        Tangential point on the line
    """
    intersection = current_line.buffer(approximation_correction).intersection(circle)

    if isinstance(intersection, Point):
        return intersection
    if isinstance(intersection, LineString):
        return current_line.interpolate(current_line.project(intersection.interpolate(0.5, True)))
    if isinstance(intersection, MultiLineString):
        closest_section = min(intersection.geoms, key=lambda section: current_line.project(section.interpolate(0.5, True)))
        return current_line.interpolate(current_line.project(closest_section.interpolate(0.5, True)))
    return None


def first_intersection(current_line: LineString, intersector: LineString) -> Point | tuple[float] | None:
    """Return the first point on the current line that lies in the intersection with the intersector.

    Args:
        current_line: The line that gets intersected. First refers to the direction of this line
        intersector: The geometry that intersects the line

    Returns
    -------
        First point on the line lying in the intersection.
    """
    intersection = current_line.intersection(intersector)

    if intersection.is_empty:
        return None
    if isinstance(intersection, Point):
        return intersection
    if isinstance(intersection, MultiPoint):
        intersection_points = list(intersection.geoms)
        intersection_points.sort(key=current_line.project)
        return intersection_points[0]

    return intersection.coords[0]


def relative_orientation(first_line: LineString, second_line: LineString, radius: float) -> int:
    """Take two lines and decide the relative orientation.

    i.e. if a turn around a circle with the given radius from the first line to the second line is a left or a right turn.
    If no such turn exists an Error is raised.

    Args:
        first_line: the first line
        second_line: the second line

    Returns
    -------
        integer with 1 corresponds to right turn and -1 corresponds to left turn.

    Raises
    ------
        ValueError: if no turn is possible
    """
    possible_intersection_directions = [
        direction for direction in [-1, 1]
        if first_intersection(
            first_line.offset_curve(direction * radius),
            second_line.offset_curve(direction * radius)) is not None
        ]

    if not possible_intersection_directions:
        raise ValueError("No turns possible!")

    if len(possible_intersection_directions) == 1:
        return possible_intersection_directions[0]

    direction = possible_intersection_directions[0]
    if isinstance(first_line, MultiLineString):
        first_line = LineString([coord for line in first_line.geoms for coord in line.coords])
    if isinstance(second_line, MultiLineString):
        second_line = LineString([coord for line in second_line.geoms for coord in line.coords])
    first_line = first_line.offset_curve(direction * radius)
    second_line = second_line.offset_curve(direction * radius)
    intersection_point = first_intersection(first_line, second_line)
    if isinstance(first_line, MultiLineString):
        first_line = LineString([coord for line in first_line.geoms for coord in line.coords])
    if isinstance(second_line, MultiLineString):
        second_line = LineString([coord for line in second_line.geoms for coord in line.coords])
    direction_point = first_intersection(first_line, second_line.offset_curve(0.01))
    if direction_point is None:
        return possible_intersection_directions[1]

    return np.sign(first_line.project(intersection_point) - first_line.project(direction_point))


def multi_poly_to_relevant_poly(mpoly: MultiPolygon, line: LineString) -> Polygon:
    """
    Extract a single polygon from a MultiPolygon that is relevant to a given line.

    Finds polygons in the MultiPolygon that are touching (within 1e-4 distance)
    the line. If multiple polygons touch the line, their union's convex hull is returned.
    If a single polygon touches the line, that polygon is returned. If no Polygon is
    touching the line, falls back to the closest Polygon

    Parameters
    ----------
    mpoly : MultiPolygon
        A collection of polygons to filter.
    line : LineString
        The reference line to find nearby polygons.

    Returns
    -------
    Polygon
        A single polygon that touches the line. Either the single touching polygon,
        or the convex hull of the union of multiple touching polygons.
    """
    if isinstance(mpoly, MultiPolygon):
        touching = [poly for poly in mpoly.geoms if poly.distance(line) < 1e-4]
        if len(touching) > 1:
            single_poly = unary_union(touching).convex_hull
        elif len(touching) == 1:
            single_poly = touching[0]
        else:
            single_poly = min(mpoly.geoms, key=lambda poly: poly.distance(line))
    return single_poly


def split_straight_endings(
        path: LineString,
        ref_line: LineString,
        ang_tol_degree: float = 0.01,
        dist_threshold: float = 1.0) -> Tuple[LineString, LineString, LineString]:
    """Splits at the first curve and the last.

    Uses an angle over 1 degree as the limit for a curve. If no curve is detected returns
    the path and two None's

    The ref_line ensures that the curve starts and begins at the ref_line or the extension of it
    """
    if len(path.coords) < 3:
        return None, path, None
    curve_start = None
    curve_end = None

    path = simplify(path, 1e-4)

    # Find the start of the curve and remember its index

    tolerance_straight = ang_tol_degree * pi / 180
    ext_ref_line = extend_line(ref_line, 10 * path.length)

    def check_curve_start(l1: LineString, l2: LineString) -> bool:
        is_curve = abs(angle_between_lines(l1, l2)) > tolerance_straight
        is_on_ref = ext_ref_line.distance(l1) < dist_threshold
        # line might be in the opposite direction to the ref_line
        is_parallel_to_ref = (abs(angle_between_lines(l1, ref_line)) < tolerance_straight
                              or abs(angle_between_lines(l1.reverse(), ref_line)) < tolerance_straight)

        return is_curve and is_on_ref and is_parallel_to_ref

    curve_start_idx = None
    for i, (c1, c2, c3) in enumerate(zip(path.coords, path.coords[1:], path.coords[2:], strict=False)):
        l1 = LineString([c1, c2])
        l2 = LineString([c2, c3])

        if check_curve_start(l1, l2):
            curve_start = path.project(Point(c2))
            curve_start_idx = i + 1  # c2 is at index i+1 in the original coords
            break
    if curve_start is None:
        return path, None, None

    # Find the first curve from the other side
    if curve_start_idx is not None:
        # create loop over coordinate triplets from the end of the path to curve_start
        j = len(path.coords) - curve_start_idx
        for c1, c2, c3 in zip(path.coords[-1:-j:-1], path.coords[-2:-j - 1:-1], path.coords[-3:-j - 2:-1], strict=False):
            l1 = LineString([c1, c2])
            l2 = LineString([c2, c3])

            if check_curve_start(l1, l2):
                curve_end = path.project(Point(c2))
                break

    # Fallback
    if curve_end is None:
        curve_end = path.length

    sd = 1e-4  # distance to prevent an overlap
    path_start = substring(path, 0, max(curve_start - sd, 0))
    path_curve = substring(path, curve_start, curve_end)
    path_rest = substring(path, min(curve_end + sd, path.length), path.length)
    return path_start, path_curve, path_rest


def orthogonal_distance(line1: LineString, line2: LineString) -> float:
    """Calculates the distance in orthogonal direction from line1 to line2.

    Calculates the orthogonal vector and the vector between the two nearest points.
    The dot product results in the orthogonal distance.
    """
    p1, p2 = nearest_points(line1, line2)
    line1_direction = direction_of_line(line1)
    between_vector = (p2.x - p1.x, p2.y - p1.y)

    orthogonal = (-line1_direction[1], line1_direction[0])
    orthogonal_dist = abs(between_vector[0] * orthogonal[0] + between_vector[1] * orthogonal[1])

    return orthogonal_dist
