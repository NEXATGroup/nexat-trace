from itertools import tee
from math import pi
from typing import Iterable, Iterator, List, Tuple, TypeVar

import dubins
import numpy as np
from numpy import typing as npt
from shapely import LinearRing, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, remove_repeated_points
from shapely import LinearRing, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, remove_repeated_points
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


def erode_linearring(ring: LinearRing, radius) -> LinearRing:
    """
    Smooths the linear ring to be drivable for the vehicle.

    Erodes the given linearring by buffering outwards once, inwards twice and outwards once again
    to smooth all vertices in the geometry to the given radius.

    """
    buffer = Polygon(
        ring.buffer(
            1 * radius,
            join_style="round",
            resolution=45
        ).exterior
    )
    buffer = buffer.buffer(
        -2 * radius,
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
        extend_start_line: bool = True) -> LineString | None:
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
    while current_line.intersects(target_line_extended):
        target_line_extended = substring(target_line_extended, 0.05, target_line_extended.length)

    if target_line.coords[0] != target_line.coords[-1]:
        target_front_extended = extend_line_in_bounds(target_line_extended, bounds, extend_front=True, extend_back=False)

        while current_line.intersects(target_front_extended):
            target_front_extended = substring(target_front_extended, 0, target_front_extended.length * 0.95)

        target_line_extended = target_front_extended
        target_line_extended = extend_line_in_bounds(target_line_extended, bounds, extend_front=False, extend_back=True)

    intersection_point_line = current_line_extended.intersection(target_line_extended)
    current_point_projection = current_line_extended.project(current_point)
    nearest_intersection_point_to_point = None
    if isinstance(intersection_point_line, Point):
        nearest_intersection_point_to_point = intersection_point_line

    elif isinstance(intersection_point_line, MultiPoint):
        intersection_point_line_list = []
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
    for intersection_point in intersection_points_buffer.geoms:
        ipol_current = current_line_extended.interpolate(current_line_extended.project(intersection_point))
        ipol_target = target_line_extended.interpolate(target_line_extended.project(intersection_point))

        if (current_line_extended.project(ipol_current)
                < current_line_extended.project(nearest_intersection_point_to_point)
                and target_line_extended.project(ipol_target)
                > target_line_extended.project(nearest_intersection_point_to_point)
                and current_line_extended.project(ipol_current) - current_point_projection > 0
                or intersection_point_line.is_empty):
            ipol_candidates.append(
                (
                    current_line_extended.project(ipol_current) - current_point_projection,
                    intersection_point,
                    ipol_current,
                    ipol_target,
                )
            )
    ipol_candidates.sort(key=lambda x: x[0])
    if ipol_candidates == []:
        return None

    _, intersection_point, ipol_current, ipol_target = ipol_candidates.pop(0)

    current_segment_new_coords = substring(
        current_line_extended,
        0.0,
        current_line_extended.project(ipol_current)
    ).coords[-2:]

    if len(current_segment_new_coords) < 2:
        return None

    current_segment = LineString(current_segment_new_coords)

    target_segment = LineString(
        substring(
            target_line_extended,
            target_line_extended.project(ipol_target),
            target_line_extended.length
        ).coords[:2]
    )
    current_vector = LineString((current_segment.boundary.geoms[0], ipol_current))
    target_vector = LineString((ipol_target, target_segment.boundary.geoms[-1]))
    if abs(angle_between_lines(current_vector, target_vector)) < pi / 4:
        turning_radius -= 0.5
    path = dubins_between_vectors(current_vector, target_vector, turning_radius)
    return path


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
    line2_end = line2.interpolate(1.0, True)
    if line1_offset_l.centroid.distance(line2_end) < line1_offset_r.centroid.distance(line2_end):
        line1_offset = line1_offset_l

    line1_start = line1.interpolate(0.0)
    line2_offset_r = line2.parallel_offset(turning_radius)
    line2_offset_l = line2.parallel_offset(-turning_radius)
    line2_offset = line2_offset_r
    if line2_offset_l.centroid.distance(line1_start) < line2_offset_r.centroid.distance(line1_start):
        line2_offset = line2_offset_l

    pivot_location: Point | None = None
    intersection = line1_offset.intersection(line2_offset)
    if isinstance(intersection, Point):
        pivot_location = intersection
    elif isinstance(intersection, MultiPoint):
        pivot_location = max(list(intersection.geoms), key = lambda p: from_segment.project(p))
    else:
        return None

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
        return None

    curve_start = LineString(
        [
            curve.coords[0],
            curve.coords[1]
        ]
    )
    if abs(angle_between_lines(curve_start, line1)) > pi / 2:
        return None

    curve_end = LineString(
        [
            curve.coords[-2],
            curve.coords[-1]
        ]
    )
    if abs(angle_between_lines(curve_end, line2)) > pi / 2:
        return None

    return curve


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
