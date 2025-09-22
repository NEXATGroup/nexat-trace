# Nexat Terrain Routing And Coverage Engine

A sophisticated complete coverage path planning library developed for controlled traffic farming applications. Specifically useful for vehicles with nonholonomic steering kinematics that have a similar turning radius to its track / working width.


## Features

### Complete coverage path planning

This library excels in robust & intelligent route optimization and curve planning for complex field geometries.

[video of TRACE example]

### Flexible route / task specification

The route planner has a lot of options and parameters that change the way the route is planned and how the curves are calculated.

[image of route with working width]

Planner parameters include:
- Start / finish location
- Variable working width (multiple of tack width e.g. for spraying applications)
- Block working configuration (group sets of neighboring ab lines together)
- Reusing existing paths on a track system to minimize soil compaction
- Working corridor error avoidance
- Multiple turning maneuvers
- Weighted prioritization of overall distance vs overall coverage

## Installation

Releases of this library are hosted on pip

```
pip install nexat-trace
```


## Usage

When using this library, you should start with generating a track system. The ```TrackSystem``` class provides basic track system generation from an outer field border:

> [!NOTE] All geometry should be in a metric coordinate system e.g. UTM projection.

```python
from nexat_trace import TrackSystem

track_system = TrackSystem.from_border(
    field_border,  # your outer field border as a shapely Polygon with holes as obstacles
    14.0,  # desired track width in meters
    reference_ab_line,  # reference LineString within your field border
    [0.5, 1.0, 1.0, 0.5]  # headland widths configuration
)
```

To use this library most effectively, you should generate your own specific track systems with field border, headlands, obstacles, AB lines and obstacle avoidance segments.

Now it is time to configure the planner parameters to match the desired task definition. Here is a basic example:

```python
from nexat_trace import RoutePlanner, CorridorStrategy

planner = RoutePlanner()

# should be whole multiple of track width
planner.route_params.working_width = 14.0

# ignore working corridor errors for now
planner.route_params.corridor_strategy = CorridorStrategy.DRIVE_NONE

# neutral distance optimizing weights
planner.route_params.weights.headland_distance_factor = 1.0
planner.route_params.weights.headland_cost_exponent = 1.0
```

Now a route can be planned using the prepared track system and configured planner:

```python
route = planner.plan_route_from_track_system(
    track_system,  # prepared track system instance
    5,  # time in s spent doing guided local search optimization
)
path = route.get_linestring()
```

The route can be plotted using the utility functions:
> [!NOTE] For this you need to have the dev requirements installed. See dev_requirements.txt or setup_venv.sh for info

```python
from nexat_trace.util import plot_geometry as pg
pg.plot_linestring_rainbow(path)
pg.show_plot()
```


## Developing

When developing on linux you can use 
```bash
source setup_venv.sh
```
to setup & activate a python venv with all development dependencies. The script also builds and installs the pydubins extension.

Now files in the root of the repo like ```example_basic.py``` or ```example_complex.py``` can be ran and debugged running code in this repo.


## Credits

This library depends on [shapely](https://github.com/shapely/shapely), [ortools](https://developers.google.com/optimization) and [numpy](https://numpy.org/) as well as [pydubins](https://github.com/rdesc/pydubins).

The pydubins module is redistributed in the nexat-trace package. See THIRD_PARTY file for license info of the pydubins software.
