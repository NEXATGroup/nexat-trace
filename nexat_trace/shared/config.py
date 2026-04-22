import json
from enum import Enum
from typing import List, Tuple

from shapely import LineString, Point

from nexat_trace.shared.curve import CurveType
from nexat_trace.shared.weights import Weights


class CorridorStrategy(Enum):
    """
    Enum for working corridor avoidance strategy.
    """

    DRIVE_ALL = 0
    DRIVE_ONLY_OUTER_HEADLAND = 1
    DRIVE_NONE = 2


class PostSteps(Enum):
    """
    Enum for post processing step procedures.
    """

    CUTOUT_AVOIDANCE = 0        # runs first
    AB_LINE_INTERPOLATION = 1   # runs last


class RoutePlanningConfig:
    """
    Holds route planning configuration and parameters.

    Route Parameters
    ----------------

    - working_width:
        Working width of implement in meters
    - weights:
        Graph cost weights, see Weights class for more info
    - neighbor_curve_distance_multiplier:
        Edge cost multiplier for neighbor curves (pi / k turn maneuver)
    - max_block_size:
        Upper limit for AB blocks - set to 0 to disable block prioritization
    - min_block_size:
        Lower limit for AB blocks
    - driven_paths:
        List of LineStrings that represent already existing paths on the given track system. When a path is given, the track
        graph marks edges that are not taken by the given paths and adds Weights.missed_path_penalty during traversal cost
        calculation. This may help reduce soil compaction with multiple paths on the same field.
    - symmetric_turn_tracks:
        Wether or not to treat the Paths in self.driven_paths omnidirectional. If this is False, an edge that is taken by a
        given existing path in the forwards direction, will be penalized during the cost calculation in the backwards
        direction.
    - corridor_strategy:
        When to fill working corridors completely
    - corridor_threshold:
        Threshold in meters above which a working corridor end is considered problematic during curve planning
    - implement_working_offset:
        Distance in meters behind the machine center, where the implement is working also known as boom. This is used to calculate
        the working corridor and relevant for the working corridor error.
    - corridor_curve:
        Determines how curves are driven if there is a problematic working corridor
    - loop_curve_initial_extend:
        The distance in meters that is extended on headland on first part of loop curve
    - loop_curve_lookahead:
        The distance in meters between the first half turn of a loop curve and the connection to the target headland
    - override_headland_index:
        Override index of target turning headland. If None the target headland will be determined automatically.
    - fully_circle_cutouts:
        Wether or not to drive around every cutout once on first encounter on route
    - round_trip_route:
        Wether or not the route optimization should see the route as a round trip from start point back to start point
        instead of some other end point
    - starting_point:
        Starting point for the route as Point in utm coords or node id. If no start or end point is given the route will lead
        from nodes nearest to top left to bottom right corner of track system bounding box.
    - finish_point:
        finish point for the route. Will only be used if round_trip_route is False. As Point in utm coords or node id.
    - working_mask_start:
        Point in utm coords or node id. If given, starting point determines the mask of ab lines for working width > track
        width. Working width mask will be determined automatically if None by maximum overlap of coverage vs least ab lines
        driven.
    - post_processing_steps:
        Dictionary mapping of post processing step and boolean value wether or not to run step after route construction
    - robust_curve_calculation_only:
        Wether or not to only use robust curve calculation functions (runs slower)

    Other Parameters
    ----------------

    - vehicle_turning_radius:
        The turning radius that will be used for the curve calculation. Also has an effect on graph construction during
        drivability checks of edges.
    - vehicle_speed_straight:
        Approximate vehicle speed on a straight path in m/s. Used only for calculating the time a route could take.
    - vehicle_speed_curve:
        Approximate vehicle speed on a curved path in m/s. Used only for calculating the time a route could take.
    - delay_on_direction_change:
        Time in s that is added to time calculation on a driving direction change
    - speed_curve_angle_threshold:
        Angle value in rad above which a path segment is considered a 'curve' during vehicle driving time calculation.
    - direction_change_extension_distance:
        Distance in m that is appended to a path segment in a straight line on a direction change maneuver.
        Used for the vehicle to get some space to steer
    - working_corridor_extension:
        Boolean indicating whether direction change extension should be applied to working corridor calculation.
    - direct_curve_link_distance:
        Distance in m below which two primary nodes allow a direct edge via neighbor (pi / k) curve.
    - min_ab_line_length:
        Length in m below which ab lines on the track system are discarded
    - disable_pi_curves:
        Disables pi / k turn maneuvers
    - heuristic_corridor_angle
        Threshold angle (fraction of π rad: 0.5π = 90°) beyond which a curve is considered too sharp
        for the vehicle to pass without triggering a working corridor error.

    Debugging Parameters
    --------------------

    - debug_prints:
        Wether or not the planner should print debug information
    - debug_plot_field:
        Wether or not the planner should plot the track system. If True, plots the track system to the global canvas.
        Does not open the plotting window on its own. Will show up only on next plot_geometry.show_plot() call if canvas was not
        cleared manually in the meantime.
    - debug_plot_track_graph:
        Wether or not the planner should plot the field graph before route optimization

    Private members
    -------------

    - _track_width:
        Is set automatically
    - _default_enabled_post_processing_steps:
        Controls which steps run per default
    """

    def __init__(self):

        # --------------- VEHICLE PARAMETERS --------------- #

        # Minimal turning radius of vehicle
        self.vehicle_turning_radius = 13.5  # m

        # Working width of implement
        self.working_width: float = 14.0

        # ---------------- TASK PARAMETERS ----------------- #

        # Track width of the field is determined automatically
        self._track_width: float | None = None

        #
        # ---------- OPTIMIZATION COST PARAMETERS ---------- #
        #

        # Graph cost weights
        self.weights = Weights.from_input(
            turn_cost_gain=1,
            turn_cost_exponent=1,
            turn_cost_angle_exponent=0,
            headland_distance_factor=1.5,
            headland_cost_exponent=1,
            corridor_error_cost=200.0,
            global_cost_offset=0.0,
            global_cost_gain=1.0
        )

        # Cost multiplier for neighbor curves
        self.neighbor_curve_distance_multiplier: float = 4.5

        # block config
        self.max_block_size: int = 9
        self.min_block_size: int = 5

        # reusing of wheel tracks config

        # any paths that were already driven on the field
        # for each path the graph edges of the turns to headland will receive weighted cost reduction
        # This way tracks in the soil can be reused by new routes
        self.driven_paths: List[LineString] = []

        # If the resulting wheel tracks of a path will be the same when driven the other way around
        # the bonuses for driven path edges will be applied in both directions and not just the way
        # the routes in self.driven_paths are oriented
        self.symmetric_turn_tracks: bool = False

        #
        # -------------- STRATEGY PARAMETERS & CURVE GEOMETRY --------------- #
        #

        # when to fill working corridors completely
        self.corridor_strategy: CorridorStrategy = CorridorStrategy.DRIVE_ONLY_OUTER_HEADLAND
        # Threshold in meters above which a working corridor end is considered problematic during curve planning
        self.corridor_threshold: float = 1.0
        # Offset of the implement from the center of the machine in m. This is used to calculate the working corridor and relevant for the working corridor error.
        self.implement_working_offset: float = 0.0
        # determines how curves are driven if there is a problematic working corridor
        self.corridor_curve: CurveType = CurveType.HOOK

        # the distance that is extended on headland on first part of loop curve in m
        self.loop_curve_initial_extend: float = 8.0
        # the distance between the half turn and the connection to the target headland in m
        self.loop_curve_lookahead: float = 28.0

        # override index of target turning headland
        # if None the target headland will be determined automatically
        self.override_headland_index: int | None = None

        # wether or not to drive around every cutout once
        self.fully_circle_cutouts: bool = True

        # Wether or not the route optimization should see the route as a round trip from start point back to start point
        # instead of some other end point
        self.round_trip_route: bool = True

        # starting point for the route as Point in utm coords or node id
        self.starting_point: Point | int | None = None
        # finish point for the route will only be used if round_trip_route is False
        # as Point in utm coords or node id
        self.finish_point: Point | int | None = None

        # wether or not to use the starting point to determine the mask of ab lines for
        # working widths > track width
        # working width mask will be determined automatically if None
        self.working_mask_start: Point | int | None = None

        self.robust_curve_calculation_only: bool = True

        #
        # ----------- POSTPROCESSING PARAMETERS ------------ #
        #

        _default_enabled_post_processing_steps = [
            PostSteps.CUTOUT_AVOIDANCE
        ]

        self.post_processing_steps = {}
        for step in PostSteps:
            self.post_processing_steps[step] = step in _default_enabled_post_processing_steps
        # see post_processing.py for further information

        #
        # ---------------- DEBUG PARAMETERS ---------------- #
        #

        self.debug_prints: bool = False

        # debug plotting using matplotlib
        self.debug_plot_field: bool = False
        self.debug_plot_track_graph: bool = False

        #
        # ---------------- MISC PARAMETERS ----------------- #
        #

        # Planner config
        self.vehicle_speed_straight = 4.16  # m/s -> 15km/h
        self.vehicle_speed_curve = 2.08  # m/s -> 7,5km/h
        self.speed_curve_angle_threshold = 0.0001

        self.direction_change_extension_distance = 5.0  # m
        self.working_corridor_extension = False
        self.heuristic_corridor_angle = 0.0  # fraction of π rad (0.5π = 90°)

        self.direct_curve_link_distance = 25.0  # m

        # time offset for direction change in s
        self.delay_on_direction_change = 15.0

        self.min_ab_line_length = 0.1  # m

        self.disable_pi_curves = True

    def copy(self):
        """
        Returns a deep copy of the instance.
        """
        new = RoutePlanningConfig()
        new.corridor_threshold = self.corridor_threshold
        new.implement_working_offset = self.implement_working_offset
        new.corridor_strategy = self.corridor_strategy
        new.max_block_size = self.max_block_size
        new.min_block_size = self.min_block_size
        new.neighbor_curve_distance_multiplier = self.neighbor_curve_distance_multiplier
        new.weights = self.weights.copy()
        new.vehicle_turning_radius = self.vehicle_turning_radius
        new.working_width = self.working_width
        new.corridor_curve = self.corridor_curve
        new.override_headland_index = self.override_headland_index
        new.fully_circle_cutouts = self.fully_circle_cutouts
        new.round_trip_route = self.round_trip_route
        new.starting_point = self.starting_point
        new.finish_point = self.finish_point
        new.working_mask_start = self.working_mask_start
        new.post_processing_steps = self.post_processing_steps.copy()
        new.robust_curve_calculation_only = self.robust_curve_calculation_only

        new._track_width = self._track_width

        new.debug_prints = self.debug_prints
        new.debug_plot_field = self.debug_plot_field
        new.debug_plot_track_graph = self.debug_plot_track_graph
        new.vehicle_speed_straight = self.vehicle_speed_straight
        new.vehicle_speed_curve = self.vehicle_speed_curve
        new.speed_curve_angle_threshold = self.speed_curve_angle_threshold
        new.direction_change_extension_distance = self.direction_change_extension_distance
        new.working_corridor_extension = self.working_corridor_extension
        new.heuristic_corridor_angle = self.heuristic_corridor_angle
        new.direct_curve_link_distance = self.direct_curve_link_distance
        new.delay_on_direction_change = self.delay_on_direction_change
        new.min_ab_line_length = self.min_ab_line_length
        new.disable_pi_curves = self.disable_pi_curves

        return new

    def __str__(self):
        """
        Returns a string representation of the instance.
        """
        return (
            f"RoutePlanningConfig(\n"
            f"  working_width={self.working_width},\n"
            f"  weights={str(self.weights)},\n"
            f"  neighbor_curve_distance_multiplier={self.neighbor_curve_distance_multiplier},\n"
            f"  max_block_size={self.max_block_size},\n"
            f"  min_block_size={self.min_block_size},\n"
            f"  corridor_strategy={self.corridor_strategy},\n"
            f"  corridor_curve={self.corridor_curve},\n"
            f"  corridor_threshold={self.corridor_threshold},\n"
            f" implement_working_offset={self.implement_working_offset},\n"
            f"  override_headland_index={self.override_headland_index},\n"
            f"  fully_circle_cutouts={self.fully_circle_cutouts},\n"
            f"  round_trip_route={self.round_trip_route},\n"
            f"  starting_point={self.starting_point},\n"
            f"  finish_point={self.finish_point},\n"
            f"  working_mask_start={self.working_mask_start},\n"
            f"  post_processing_steps={self.post_processing_steps},\n"
            f"  robust_curve_calculation_only={self.robust_curve_calculation_only},\n"
            f"  heuristic_corridor_angle={self.heuristic_corridor_angle}\n"

            f"  _track_width={self._track_width},\n"

            f"  debug_prints={self.debug_prints},\n"
            f"  debug_plot_field={self.debug_plot_field},\n"
            f"  debug_plot_track_graph={self.debug_plot_track_graph},\n"
            f")"
        )

    def to_json(self) -> str:
        """
        Returns a serialized json string of the config.
        """
        def serialize(obj):
            if isinstance(obj, (CorridorStrategy, CurveType, PostSteps)):
                return obj.name
            if isinstance(obj, Point):
                return (obj.x, obj.y)
            if isinstance(obj, Weights):
                return obj.__dict__
            if isinstance(obj, dict):
                return {serialize(k): serialize(v) for k, v in obj.items()}
            if isinstance(obj, set):
                return list(obj)
            return obj

        data = {}
        for key in self.__dict__:
            print(key)
            data[key] = serialize(self.__dict__[key])
        print(data)
        return json.dumps(data)

    @staticmethod
    def from_json(input_data: str | dict):
        """
        Makes an instance from json data.
        """
        if isinstance(input_data, str):
            data = json.loads(input_data)
        elif isinstance(input_data, dict):
            data = input_data
        else:
            raise TypeError(f"Cannot parse RoutePlanningConfig from {type(input_data)}")

        new = RoutePlanningConfig()

        def get_location(location: int | Tuple[float, float] | None):
            if location is None:
                return None

            if isinstance(location, int):
                return location
            elif isinstance(location, tuple):
                p = Point(location)
                return p

        new.working_width = data["working_width"]
        new.weights = Weights.from_input(
            data["weights"]["turn_cost_gain"],
            data["weights"]["turn_cost_exponent"],
            data["weights"]["turn_cost_angle_exponent"],
            data["weights"]["headland_distance_factor"],
            data["weights"]["headland_cost_exponent"],
            data["weights"]["corridor_error_cost"],
            data["weights"]["missed_path_penalty"],
            data["weights"]["different_block_penalty"],
            data["weights"]["global_cost_offset"],
            data["weights"]["global_cost_gain"]
        )
        new.neighbor_curve_distance_multiplier = data["neighbor_curve_distance_multiplier"]
        new.max_block_size = data["max_block_size"]
        new.min_block_size = data["min_block_size"]
        new.corridor_strategy = CorridorStrategy[data["corridor_strategy"]]
        new.corridor_curve = CurveType[data["corridor_curve"]]
        new.corridor_threshold = data["corridor_threshold"]
        new.implement_working_offset = data["implement_working_offset"]
        new.override_headland_index = data["override_headland_index"]
        new.fully_circle_cutouts = data["fully_circle_cutouts"]
        new.round_trip_route = data["round_trip_route"]
        new.starting_point = get_location(data["starting_point"])
        new.finish_point = get_location(data["finish_point"])
        new.working_mask_start = get_location(data["working_mask_start"])
        for entry in data["post_processing_steps"].items():
            new.post_processing_steps[PostSteps[entry[0]]] = bool(entry[1])
        new.robust_curve_calculation_only = data["robust_curve_calculation_only"]
        new.heuristic_corridor_angle = data["heuristic_corridor_angle"]
        new.debug_plot_field = data["debug_plot_field"]
        new.debug_plot_track_graph = data["debug_plot_track_graph"]
        new.debug_prints = data["debug_prints"]

        return new
