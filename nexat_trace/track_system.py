import hashlib
import json
from typing import List

from shapely import LinearRing, LineString, MultiLineString, Polygon
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from nexat_trace.util import geom_tools as gt


class Feature:
    """
    GeoJSON feature data class.
    """

    def __init__(self, geometry: BaseGeometry | dict, properties: dict | None = None):
        self.geometry = {}
        if isinstance(geometry, BaseGeometry):
            self.geometry: dict = mapping(geometry)
        elif isinstance(geometry, dict):
            self.geometry = geometry
        else:
            raise TypeError("Feature only supports BaseGeometry or dict data")
        self.properties: dict = properties or {}

    def to_dict(self) -> dict:
        """
        Returns the feature as a dict.
        """
        return {
            "type": "Feature",
            "properties": self.properties,
            "geometry": self.geometry
        }

    @staticmethod
    def from_dict(data: dict):
        """
        Parses instance from GeoJSON representation dict.
        """
        return Feature(
            data["geometry"],
            data["properties"]
        )


class FeatureCollection:
    """
    GeoJson FeatureCollection data class.
    """

    def __init__(self, features: List[Feature]):
        self.type: str = "FeatureCollection"
        self.features: List[Feature] = features

    def to_dict(self) -> dict:
        """
        Returns the FeatureCollection as a dict.
        """
        return {
            "type": "FeatureCollection",
            "features": [feature.to_dict() for feature in self.features]
        }


GEOM_TYPE_KEY = "track_system_geom_type"
BORDER_PROPERTY = "outer_border"
HEADLANDS_PROPERTY = "headlands"
HEADLAND_INDEX_PROPERTY = "headland_index"
HEADLAND_CONFIG_PROPERTY = "headland_config"
AB_LINES_PROPERTY = "ab_lines"
OBSTACLE_AVOIDANCE_PROPERTY = "obstacle_avoidance_segments"
TO_BE_EVADED_PROPERTY = "to_be_evaded_obstacles"


class TrackSystem:
    """
    Class representing the track system of a field. Geometries should be given in metric coordinates (UTM/ ENU).

    Attributes
    ----------
    outer_border : Polygon
        A polygon representing the area of the field. The polygons exterior represents the hard outer edge of the field that the
        vehicle is to avoid. The polygons interiors represent obstacles on the field and are supposed to have headlands.
    headlands : List[List[LinearRing]]
        List of Lists of LinearRings that represent the headland tracks of the field. The ordering of these collections must not
        be changed after generation. First level of Lists represent collections of tracks for each 'level' of
        headland. Lookup for outer headland tracks must be self.headlands[0] and inner headlands must be self.headlands[-1].
        The inner collections are supposed to be [outer_ring_track, cutout_1_track, cutout_2_track, ...].
    ab_lines : MultiLineString
        MultiLineString representing the AB Line tracks of the inner field. AB Lines should not intersect with any headland track.
        Using AB lines with more than 2 vertices may lead to problems. All AB Lines should be spaced an equal amount of distance
        from each other and run in parallel direction.
    headland_config : List[float]
        List of track width multipliers that indicate the configuration of the headlands. [1.0, 1.0, 0.5] would represent a
        headland configuration of three headlands where the first two are exactly one track width apart and the last 0.5 * track
        width.
    obstacle_avoidance_segments: List[LineString | LinearRing]
        List of LineStrings and/or LinearRings representing tracks that are supposed to be driven by the vehicle inside the field
        to avoid obstacles without dedicated headland tracks. These segments have to be rounded to the desired turning radius.
    to_be_evaded_obstacles: List[LineString | LinearRing]
        List of Polygons representing obstacles, that should be avoided using and evasion move, which is
        basically a hook onto a segment to avoid the obstacle with enough clearance.

    Examples
    --------

    ```python
    from nexat_trace import TrackSystem

    # from manual geometry generation
    ts = TrackSystem(
        field_border,
        headlands,
        ab_lines,
        headland_config,
        obstacle_avoidance_segments
    )

    # alternatively basic generation only from outer border and reference ab line
    ts = TrackSystem.from_border(
        field_shape,  # shapely polygon with holes as obstacles
        track_width,  # desired track width in m
        reference_line,  # reference ab line within the field shape polygon
        headland_config,  # desired headland configuration
    )

    # generate geo json representation
    data = ts.to_dict()

    # parse from geo json representation as dict or str
    ts = TrackSystem.from_geo_json(data)
    ```
    """

    def __init__(
            self,
            outer_border: Polygon,
            headlands: List[List[LinearRing]],
            ab_lines: MultiLineString,
            headland_config: List[float],
            obstacle_avoidance_segments: List[LineString | LinearRing] | None = None,
            to_be_evaded_obstacles:  List[LineString | LinearRing] | None = None):

        self.outer_border: Polygon = outer_border
        self.headlands: List[List[LinearRing]] = headlands
        self.ab_lines: MultiLineString = ab_lines
        self.headland_config: List[float] = headland_config

        if obstacle_avoidance_segments is None:
            obstacle_avoidance_segments = []
        self.obstacle_avoidance_segments: List[LineString | LinearRing] = obstacle_avoidance_segments

        if to_be_evaded_obstacles is None:
            to_be_evaded_obstacles = []
        self.to_be_evaded_obstacles: List[LineString | LinearRing] = to_be_evaded_obstacles

        all_headlands = []
        for ring_list in headlands:
            all_headlands.extend(ring_list)

        multi_headland = MultiLineString(all_headlands)
        multi_obstacle_segments = MultiLineString(obstacle_avoidance_segments)
        multi_evasion_segments = MultiLineString(to_be_evaded_obstacles)

        str_to_hash = (
            outer_border.wkt
            + multi_headland.wkt
            + ab_lines.wkt
            + str(headland_config)
            + multi_obstacle_segments.wkt
            + multi_evasion_segments.wkt
        )
        self._hash_id = hashlib.sha256(str_to_hash.encode('utf-8')).hexdigest()

    def to_dict(self) -> dict:
        """
        Returns a GeoJSON conform dict representation of the track system.
        """
        features = [
            Feature(
                self.outer_border,
                {GEOM_TYPE_KEY: BORDER_PROPERTY}
            ),
            Feature(
                self.ab_lines,
                {GEOM_TYPE_KEY: AB_LINES_PROPERTY}
            ),
            Feature(
                MultiLineString(self.obstacle_avoidance_segments),
                {GEOM_TYPE_KEY: OBSTACLE_AVOIDANCE_PROPERTY}
            ),
            Feature(
                MultiLineString(self.to_be_evaded_obstacles),
                {GEOM_TYPE_KEY: TO_BE_EVADED_PROPERTY}
            )
        ]

        for index, headland in enumerate(self.headlands):
            properties = {
                GEOM_TYPE_KEY: HEADLANDS_PROPERTY,
                HEADLAND_INDEX_PROPERTY: index,
                HEADLAND_CONFIG_PROPERTY: self.headland_config[index]
            }
            features.append(
                Feature(MultiLineString(headland), properties)
            )

        collection = FeatureCollection(features)
        return collection.to_dict()

    @staticmethod
    def from_border(
            outer_border: Polygon,
            track_width: float,
            reference_ab_line: LineString,
            headland_config: List[float],
            obstacle_avoidance_segments: List[LineString | LinearRing] | None = None,
            to_be_evaded_obstacles: List[LineString | LinearRing] | None = None,
            offset_from_border: float = 0.4):
        """
        Generates a track system from a border, field and vehicle parameters.

        Parameters
        ----------
        outer_border : Polygon
            Field area in polygon representation
        track_width : float
            Vehicle track width
        reference_ab_line : LineString
            Line from which the ab line grid will be generated. Has to be located within the outer border and has to have exactly
            2 points.
        headland_config : List[float]
            List of track width multipliers that indicate the configuration of the headlands. [1.0, 1.0, 0.5] would represent a
            headland configuration of three headlands where the first two are exactly one track width apart and the last 0.5 *
            track width.
        obstacle_avoidance_segments : List[LineString | LinearRing]
            List of LineStrings and/or LinearRings representing tracks that are supposed to be driven by the vehicle inside the
            field to avoid obstacles without dedicated headland tracks. These segments have to be rounded to the desired turning
            radius.
        to_be_evaded_obstacles : List[LineString | LinearRing]
            List of Polygons representing tracks that are supposed to be driven by the vehicle inside the
            field to avoid obstacles without dedicated headland tracks. These segments have to be rounded to the desired turning
            radius.

        Returns
        -------
            New instance of TrackSystem
        """

        # generate headlands
        inner_headland: Polygon | None = None
        sum_offset = offset_from_border
        headlands = []
        for factor in headland_config:
            buffer_distance = sum_offset - (factor * track_width)

            headland_poly = outer_border.buffer(buffer_distance, resolution=20, cap_style=2, join_style=2)
            inner_headland = headland_poly
            if not isinstance(headland_poly, Polygon):
                # TODO
                raise NotImplementedError("Multiple sub fields not yet supported")

            headlands.append([headland_poly.exterior] + list(headland_poly.interiors))

            sum_offset -= (factor * track_width)

        # generate ab lines
        if len(reference_ab_line.coords) != 2:
            raise NotImplementedError("Track system generation for multi segment ab lines not supported yet")

        if not Polygon(outer_border.exterior).contains(reference_ab_line):
            raise ValueError("Reference ab line must be located within the outer border")

        start_line = gt.extend_line(reference_ab_line, outer_border.exterior.length)

        # worst case distance to expand grid is border circumference / 2
        expand_count = round((outer_border.exterior.length / 2.0) / track_width) + 1

        line_grid: List[LineString] = [start_line]

        for i in range(1, expand_count):
            offset_distance = track_width * i
            line_grid.append(start_line.parallel_offset(offset_distance))
            line_grid.append(start_line.parallel_offset(-offset_distance))

        multi_grid = MultiLineString(line_grid)
        multi_grid = multi_grid.intersection(inner_headland)

        track_system = TrackSystem(
            outer_border, headlands, multi_grid, headland_config, obstacle_avoidance_segments, to_be_evaded_obstacles
            )

        return track_system

    @staticmethod
    def from_geo_json(geo_json: str | dict):
        """
        Parses GeoJSON data and makes a new instance of TrackSystem.

        The features in the feature collection have to have the corresponding properties set for this to be able to parse the
        track system geometries correctly. Feature properties have to match the structure of the TrackSystem.to_dict() GeoJSON
        representation.

        Parameters
        ----------

        goe_json : str | dict
            GeoJSON data in string or dictionary representation

        Returns
        -------

            Instance of TrackSystem
        """
        data: dict | None = None
        if isinstance(geo_json, dict):
            data = geo_json
        elif isinstance(geo_json, str):
            data = json.loads(geo_json)
        else:
            raise TypeError("Only str and dict are supported to parse from")

        border: Polygon | None = None
        headlands = {}
        ab_lines: MultiLineString | None = None
        headland_config_entries = {}
        headland_config: list | None = None
        obstacle_avoidance_segments: list | None = None
        to_be_evaded_obstacles: list | None = None

        for feature_data in data["features"]:
            feature = Feature.from_dict(feature_data)
            track_system_geom_type: str = feature.properties[GEOM_TYPE_KEY]

            if track_system_geom_type == BORDER_PROPERTY:
                border = shape(feature.geometry)

            elif track_system_geom_type == HEADLANDS_PROPERTY:
                index = feature.properties[HEADLAND_INDEX_PROPERTY]
                headland_config_entry = feature.properties[HEADLAND_CONFIG_PROPERTY]
                multi_geom = shape(feature.geometry)
                rings = [LinearRing(line) for line in multi_geom.geoms]
                headlands[index] = rings
                headland_config_entries[index] = headland_config_entry

            elif track_system_geom_type == AB_LINES_PROPERTY:
                ab_lines = shape(feature.geometry)

            elif track_system_geom_type == OBSTACLE_AVOIDANCE_PROPERTY:
                multi_geom = shape(feature.geometry)
                obstacle_avoidance_segments = [geom for geom in multi_geom.geoms]
            elif track_system_geom_type == TO_BE_EVADED_PROPERTY:
                multi_geom = shape(feature.geometry)
                to_be_evaded_obstacles = [geom for geom in multi_geom.geoms]

        headland_config = []
        for i in range(len(headland_config_entries)):
            headland_config.append(headland_config_entries[i])

        return TrackSystem(
            border,
            headlands,
            ab_lines,
            headland_config,
            obstacle_avoidance_segments,
            to_be_evaded_obstacles
        )
