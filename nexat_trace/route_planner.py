from threading import Thread
from typing import List, NoReturn, Tuple

from shapely.geometry import LinearRing, LineString, MultiLineString, Point, Polygon

from nexat_trace.planning import post_processing, route_planning
from nexat_trace.planning.net_graph.net_graph import NetGraph
from nexat_trace.planning.route import Route
from nexat_trace.shared import progress
from nexat_trace.shared.config import CorridorStrategy, RoutePlanningConfig
from nexat_trace.shared.exceptions import PlannerConfigurationError
from nexat_trace.shared.planning_messages import PlanningMsg
from nexat_trace.track_system import TrackSystem
from nexat_trace.util.field_conversion import (
    get_headland_index_of_path_on_track_system,
    get_working_width_of_path_on_track_system,
)


class RoutePlanner:
    """
    Class that handles graph building & planning of routes on given track systems.

    Route Parameters
    ----------------

    Planning parameters can be changed by modifying the route_params member. See RoutePlanningConfig class for more info.

    Examples
    --------

    ```python
    from nexat_trace import RoutePlanner, TrackSystem

    ts = TrackSystem.from_border(
        my_field_border,  # shapely polygon
        14.0,  # track width in m
        reference_ab_line,  # reference line within the field
        [0.5, 1.0, 1.0, 0.5]  #  headland config
    )

    planner = RoutePlanner()
    route = planner.plan_route_from_track_system(
        ts,  # prepared track system
        5  # time spent optimizing in s
    )

    path = route.get_linestring()
    ```
    """

    def __init__(self):
        self.route_params = RoutePlanningConfig()
        self.net_graph: NetGraph = None
        self.ab_lines: List[LineString] = []
        self.field_border: LinearRing = None
        self.headland_shape: LinearRing = None
        self.solver_time_limit = -1

        self._last_tracks_hash = None
        self._last_working_width = None
        self._last_headland_override = None
        self._last_working_mask_start = None
        self._last_symmetric_turn_tracks = False
        self._keep_out_polys = []
        self._route = None
        self._busy = False
        self._finished_planning = False

        self._progress = progress.PlanningProgress()
        self._occurred_exception: Exception | None = None

        self._messages = []

    def build_graphs_async(
            self,
            track_system: TrackSystem,
            graph_construction_limit: int = 1000) -> Thread | None:
        """
        Builds graphs used for planning a route in a new thread.

        Parameters
        ----------

        track_system : TrackSystem
            Prepared track system
        graph_construction_limit : int
            Limit how many neighbors are searched per primary node in track graph on net graph construction

        Returns
        -------

            Returns the instance of the new thread.
        """
        self._check_exception()
        if self._busy:
            if self.route_params.debug_prints:
                print("build_graphs_async(): ignoring call because planner is already busy")
            return None
        self._busy = True

        thread = Thread(
            target=self.build_graphs,
            args=[track_system, graph_construction_limit]
        )
        thread.start()
        return thread

    def build_graphs(
            self,
            track_system: TrackSystem,
            graph_construction_limit: int = 1000):
        """
        Builds graphs used for planning a route.

        Parameters
        ----------

        track_system : TrackSystem
            Prepared track system
        graph_construction_limit : int
            Limit how many neighbors are searched per primary node in track graph on net graph construction
        """
        self._check_exception()
        if track_system is None:
            return

        try:
            data = route_planning.net_graph_from_track_system(
                track_system,
                self.route_params,
                self._progress,
                graph_construction_limit
            )
        except Exception as e:
            self._occurred_exception = e
            raise e

        self.net_graph = data[0]
        self.ab_lines = data[1]
        self.headland_shape = data[2]
        self.field_border = data[3]
        self._messages.extend(data[4])
        self._last_working_width = self.route_params.working_width
        self._last_headland_override = self.route_params.override_headland_index
        self._last_working_mask_start = self.route_params.working_mask_start
        self._last_symmetric_turn_tracks = self.route_params.symmetric_turn_tracks
        self._busy = False

    def plan_route_from_track_system_async(
            self,
            track_system: TrackSystem,
            solver_time_limit = -1,
            graph_construction_limit = 1000) -> Thread:
        """
        Prepares graphs for a track system and plans a new route in a new thread.

        Route can be accessed when self.done() with self.get_route().

        Parameters
        ----------

        track_system : TrackSystem
            Prepared track system
        solver_time_limit: int
            Time in seconds that will be spent during guided local search optimization. Alternatively uses greedy descent
            optimization for values less than 1.
        graph_construction_limit : int
            Limit how many neighbors are searched per primary node in track graph on net graph construction

        Returns
        -------

            Returns new thread.
        """
        self._check_exception()
        self._update_config_for_async()
        if self._busy:
            if self.route_params.debug_prints:
                print("plan_route_from_track_system_async(): ignoring because planner is already busy")
            return None
        self._busy = True

        thread = Thread(
            target=self.plan_route_from_track_system,
            args=[
                track_system,
                solver_time_limit,
                graph_construction_limit
            ]
        )
        thread.start()
        return thread

    def plan_route_from_track_system(
            self,
            track_system: TrackSystem,
            solver_time_limit: int = -1,
            graph_construction_limit = 1000) -> Route | None:
        """
        Prepares a given track system and returns Route object or None.

        Automatically handles (re)building of graphs for route optimization.

        Parameters
        ----------

        track_system : TrackSystem
            Prepared track system
        solver_time_limit: int
            Time in seconds that will be spent during guided local search optimization. Alternatively uses greedy descent
            optimization for values less than 1.
        graph_construction_limit : int
            Limit how many neighbors are searched per primary node in track graph on net graph construction

        Returns
        -------

            Returns instance of new route if successful else None.
        """
        self._check_exception()
        if track_system is None:
            return None
        self._finished_planning = False

        self.load_track_system(track_system, graph_construction_limit)
        if solver_time_limit is not None:
            self.solver_time_limit = solver_time_limit
        result = self.plan_route(self.solver_time_limit)

        return result

    def plan_route_async(self, solver_time_limit = -1) -> Thread:
        """
        Plans a new route from current graphs and config.

        Route can be accessed when self.done() with self.get_route().

        Parameters
        ----------

        solver_time_limit: int
            Time in seconds that will be spent during guided local search optimization. Alternatively uses greedy descent
            optimization for values less than 1.

        Returns
        -------

            Returns new thread.
        """
        self._check_exception()
        self._update_config_for_async()
        if self._busy:
            if self.route_params.debug_prints:
                print("plan_route_async(): ignoring because planner is already busy")
            return None
        self._busy = True

        thread = Thread(
            target=self.plan_route,
            args=[solver_time_limit]
        )
        thread.start()
        return thread

    def plan_route(self, solver_time_limit = None) -> Route | None:
        """
        Plans a new route from current graphs and config.

        Parameters
        ----------

        solver_time_limit: int
            Time in seconds that will be spent during guided local search optimization. Alternatively uses greedy descent
            optimization for values less than 1.

        Returns
        -------

            Returns instance of new route if successful else None.
        """
        self._check_exception()
        self._busy = True
        self._route = None
        self._finished_planning = False

        if self.net_graph is None:
            self._busy = False
            if self.route_params.debug_prints:
                print("Did not start planning because there was no net graph loaded")
            return None

        if solver_time_limit is not None:
            self.solver_time_limit = solver_time_limit
        try:
            self._route, warnings = route_planning.solve_route(
                self.net_graph,
                self.route_params,
                self._progress,
                self.ab_lines,
                self.headland_shape,
                self.field_border,
                solver_time_limit = self.solver_time_limit
            )
            self._messages.extend(warnings)
            self._messages.extend(
                [msg for msg in self._route.planning_messages if msg not in self._messages]
            )

            self._route.planning_messages = self._messages.copy()

            self._apply_postprocessing()
        except Exception as e:
            self._occurred_exception = e
            raise e

        self._busy = False
        self._finished_planning = True
        return self._route

    def get_route(self) -> Route | None:
        """
        Returns the latest route if planning is done.
        """
        self._check_exception()
        if self.done():
            return self._route
        return None

    def done(self) -> bool:
        """
        Returns True if the planner is done planning a route.
        """
        self._check_exception()
        return self._finished_planning

    def get_progress(self) -> Tuple[int, str]:
        """
        Returns the overall planning progress in percent and current planning stage as str.
        """
        self._check_exception()
        return (self._progress.planning_percent, self._progress.planning_stage.value)

    def load_track_system(
            self,
            track_system: TrackSystem,
            graph_construction_limit = 1000):
        """
        Prepares planner for given track system.

        Parameters
        ----------

        track_system : TrackSystem
            Prepared track system
        graph_construction_limit : int
            Limit how many neighbors are searched per primary node in track graph on net graph construction
        """
        self._check_exception()
        if track_system is None:
            return
        if not self.is_loaded_track_system(track_system):
            self._progress.planning_stage = progress.PlanningStage.TRACK_GRAPH
            self._progress.planning_percent = 0
            self.build_graphs(track_system, graph_construction_limit)
            self._last_tracks_hash = track_system._hash_id

    def load_track_system_async(
            self,
            track_system: TrackSystem,
            graph_construction_limit = 1000) -> Thread:
        """
        Prepares planner for given track system in a new thread.

        Parameters
        ----------

        track_system : TrackSystem
            Prepared track system
        graph_construction_limit : int
            Limit how many neighbors are searched per primary node in track graph on net graph construction

        Returns
        -------

            Returns a new thread.
        """
        self._check_exception()
        self._update_config_for_async()
        if self._busy:
            if self.route_params.debug_prints:
                print("load_track_system_async(): ignoring because planner is already busy")
            return None
        self._busy = True

        thread = Thread(
            target=self.load_track_system,
            args=[track_system, graph_construction_limit]
        )
        thread.start()
        return thread

    def set_keep_out_areas(
            self,
            polygons: List[Polygon]):
        """
        Sets the areas that will not be covered by a planned route.

        Can only be called when a track system is loaded.
        """
        self._check_exception()
        if self.net_graph is None:
            if self.route_params.debug_prints:
                print("set_keep_out_areas() had no effect because no track system was loaded")
            return
        self._keep_out_polys = polygons.copy()
        self.net_graph.set_exclude_areas(self._keep_out_polys)

    def is_loaded_track_system(self, other_track_system: TrackSystem) -> bool:
        """
        Wether or not a given track system can be planned with the loaded graphs or not.

        Returns True if the loaded track system inside the planner is the track system with the given id and if the config is
        compatible with the loaded graphs.
        """
        self._check_exception()
        if self._last_tracks_hash is None:
            if self.route_params.debug_prints:
                print("Last track system hash was None, building graphs")
            return False
        if self._last_tracks_hash != other_track_system._hash_id:
            if self.route_params.debug_prints:
                print("Last track system hash was different, building graphs")
            return False

        if self._last_working_width != self.route_params.working_width:
            if self.route_params.debug_prints:
                print("Working width was different, building graphs")
            return False
        if self._last_headland_override != self.route_params.override_headland_index:
            if self.route_params.debug_prints:
                print("Headland override index was different, building graphs")
            return False
        if self._last_working_mask_start != self.route_params.working_mask_start:
            if self.route_params.debug_prints:
                print("Working subset override was different, building graphs")
            return False
        if self._last_symmetric_turn_tracks != self.route_params.symmetric_turn_tracks:
            if self.route_params.debug_prints:
                print("Symmetric turn tracks was different, building graphs")
            return False
        return True

    def shortest_path_from_to(
            self,
            start_geom: Point | LineString,
            target_geom: Point | LineString,
            fallback = True) -> Route | None:
        """
        Plans a path from the given start geometry to the target geometry along the current graphs.

        Parameters
        ----------
            start_geom : Point | LineString
                Start location of navigation. If LineString the last point on the line will determine the start heading of
                navigation.
            target_geom : Point | LineString
                Target location of navigation. If LineString the last point on the line will determine the target heading of
                navigation.
            fallback : bool
                If False, only edges and subroutes already known to the loaded net graph will be used to search the shortest route
                and no new edges will be explored.

        Returns
        -------

            A new Route instance.
        """
        self._check_exception()

        # temporarily disable the filling of working corridors and looping of path
        last_fill_corridors = self.route_params.corridor_strategy
        self.route_params.corridor_strategy = CorridorStrategy.DRIVE_NONE
        last_looping = self.route_params.round_trip_route
        self.route_params.round_trip_route = False

        self._route = route_planning.navigate_from_to(
            self.route_params,
            start_geom,
            target_geom,
            self.net_graph,
            fallback
        )
        if self._route.smoothing_error:
            self._messages.append(PlanningMsg.CURVE_CALCULATION_ERROR)
        self._route.planning_messages = self._messages.copy()
        self._apply_postprocessing()
        self.route_params.corridor_strategy = last_fill_corridors
        self.route_params.round_trip_route = last_looping
        return self._route

    def has_messages(self) -> bool:
        """
        Returns True if the planner has planning messages to the user.
        """
        self._check_exception()
        return len(self._messages) > 0

    def pop_messages(self) -> List[PlanningMsg]:
        """
        Returns the list of current planning messages and clears them.
        """
        self._check_exception()
        msgs = self._messages.copy()
        self._messages.clear()
        return msgs

    def set_config_of_path(
            self,
            path: LineString | MultiLineString,
            track_system: TrackSystem | None = None,
            overwrite_starting_point: bool = True,
            overwrite_finish_point: bool = True,
            overwrite_working_mask_start: bool = True,
            overwrite_working_width: bool = True,
            overwrite_target_headland: bool = True):
        """
        Copy the planner config of a given path on a track system.

        Sets working_width, starting_point, finish_point, working_mask_start and override_headland_index parameters of
        self.route_params according to the given path and loaded track system.

        May lead to rebuilding of loaded graphs if the target track system demands it.

        Parameters
        ----------
        path : LineString | MultiLineString
            The path to copy the planner configuration from
        track_system : TrackSystem | None
            The track system the path is on. If None it is assumed that the currently loaded track system is the correct one.
        overwrite_starting_point : bool
            Wether or not to copy the config for self.route_params.starting_point.
        overwrite_finish_point : bool
            Wether or not to copy the config for self.route_params.finish_point.
        overwrite_working_mask_start : bool
            Wether or not to copy the config for self.route_params.working_mask_start.
        overwrite_working_width : bool
            Wether or not to copy the config for self.route_params.working_width.
        overwrite_target_headland : bool
            Wether or not to copy the config for self.route_params.target_headland.
        """
        self._check_exception()

        if track_system is None:
            if self.net_graph is None:
                raise PlannerConfigurationError(
                    "Cannot set config by given path if no track system is loaded or passed as a parameter"
                )
            else:
                track_system = self.net_graph.track_graph.track_system

        if isinstance(path, MultiLineString):
            coords = []
            [coords.extend(segment.coords) for segment in path.geoms]
            continuous_path = LineString(coords)
        elif isinstance(path, LineString):
            continuous_path = path
        else:
            raise TypeError("Path has to be LineString or MultiLineString")

        if overwrite_starting_point:
            self.route_params.starting_point = Point(continuous_path.coords[0])

        if overwrite_working_mask_start:
            self.route_params.working_mask_start = self.route_params.starting_point

        if overwrite_finish_point:
            self.route_params.finish_point = Point(continuous_path.coords[-1])

        if overwrite_target_headland:
            headland_index = get_headland_index_of_path_on_track_system(path, track_system)
            if headland_index is None:
                raise PlannerConfigurationError("Could not find target headland of given path on track system")

            self.route_params.override_headland_index = headland_index

        if overwrite_working_width:
            self.route_params.working_width = get_working_width_of_path_on_track_system(path, track_system)

    def _apply_postprocessing(self):
        """
        Applies all selected post processing steps.

        Is executed automatically when a new route is planned.
        Do not call yourself.
        """
        if self._route is None:
            return

        for step in self.route_params.post_processing_steps:

            if not self.route_params.post_processing_steps[step]:
                continue

            if self.route_params.debug_prints:
                print(f"Applying postprocessing: {step.name}")
            post_processing.FUNCTIONS[step](self._route)

        if post_processing.collision_in_route(self._route):
            self._messages.append(PlanningMsg.COLLISION)
            self._route.planning_messages.append(PlanningMsg.COLLISION)

        post_processing.simplify_points(self._route)

        self._route._finalize()

    def _check_exception(self) -> None | NoReturn:
        """
        Raises any exception that occurred during async function calls.

        This function should be called at the top of any public function of the planner.
        """
        if self._occurred_exception is None:
            return
        self._progress.planning_percent = 100
        self._progress.planning_stage = "Path planning failed"

        # clear occurred exception of instance
        tmp = self._occurred_exception
        self._occurred_exception = None
        self._finished_planning = False
        self._busy = False
        self._route = None
        self._progress.planning_percent = 0
        raise tmp

    def _update_config_for_async(self):
        """
        Updates values in self.route_params to work with running in another thread.

        Changed Values
        --------------

        - debug_plot_field -> False
        - debug_plot_track_graph -> False
        """
        self.route_params.debug_plot_field = False
        self.route_params.debug_plot_track_graph = False
        self.route_params.debug_plot_net_graph = False
