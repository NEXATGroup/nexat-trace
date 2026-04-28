import random

import matplotlib.pyplot as plt
import matplotlib.style
from shapely import LineString, Point, Polygon

"""
This module is meant for debugging purposes only.
Can be used to plot shapely geometries with matplotlib.
"""

fig = None


def plot_linestring(line: LineString, color="black", linewidth=2, text=None, with_arrow=False):
    """
    Plots a line with set color and linewidth.

    Color can be set to 'random' for random color.
    If a text is passed it will plot the text at the middle of the line.
    """
    global fig

    if isinstance(color, str) and color == "random":
        color = get_random_color()

    if line is None:
        return

    if len(line.coords) > 2:
        for i in range(0, len(line.coords) - 1):
            p1 = line.coords[i]
            p2 = line.coords[i + 1]
            ls = LineString([p1, p2])
            if i == len(line.coords) - 1 // 2:
                plot_linestring(ls, color, linewidth, text, with_arrow)
            else:
                plot_linestring(ls, color, linewidth, None, with_arrow)
        return

    plt.plot(line.xy[0], line.xy[1], color=color, linewidth=linewidth)

    if text is not None or with_arrow:
        mid_point: Point
        mid_point = line.interpolate(0.5 + random.random() * 0.1, normalized = True)

    if with_arrow:
        dx = (line.coords[-1][0] - line.coords[0][0]) * 0.18
        dy = (line.coords[-1][1] - line.coords[0][1]) * 0.18
        plt.arrow(
            mid_point.x - dx / 2.0,
            mid_point.y - dy / 2.0,
            dx,
            dy,
            head_width=2.5,
            head_length=3.0,
            fc=color,
            ec=color,
            linewidth=linewidth
        )

    if text is not None:
        plt.text(mid_point.x, mid_point.y, text, fontsize=10)


def plot_polygon(poly: Polygon, color = "red", linewidth = 2, alpha = 0.5, text=None):
    """
    Plots a line with set color and linewidth.

    Color can be set to 'random' for random color.
    If a text is passed it will plot the text at the middle of the line.
    """
    global fig

    if isinstance(color, str) and color == "random":
        color = get_random_color()

    if poly is None:
        return

    x, y = poly.exterior.xy

    plt.plot(x, y, color=color, linewidth=linewidth)
    plt.fill(x, y, color=color, alpha=alpha)

    if text is not None:
        mid_point = poly.centroid
        plt.text(mid_point.x, mid_point.y, text, fontsize = 10)


def plot_linestring_rainbow(line, rainbow_repeats = 1, linewidth = 2, with_arrow=False):
    """
    Plots the segments of the linestring in a different hue depending on the progression of segments.
    """

    is_closed = line.geom_type == "LinearRing"
    iteration_range = range(0, len(line.coords) - 1, 1)
    if is_closed:
        iteration_range = range(-1, len(line.coords) - 1, 1)

    progress = 0.0
    for i in iteration_range:
        progress = i / len(line.coords)
        hue = 1.0 - (rainbow_repeats * progress) % 1.0
        segment_color = matplotlib.colors.hsv_to_rgb([hue, 1.0, 1.0])
        c1 = line.coords[i]
        c2 = line.coords[i + 1]
        segment = LineString([c1, c2])
        plot_linestring(segment, segment_color, linewidth, with_arrow=with_arrow)


def plot_linestring_list(lines, color = "green", linewidth = 2, text = None, with_arrow = False):
    """
    Plots a list of nodes as a path.
    """
    if isinstance(color, str) and color == "random":
        color = get_random_color()

    [plot_linestring(line, color, linewidth, text, with_arrow) for line in lines]


def plot_point(point: Point, marker = "0", markersize = 10, color = "red", text = None):
    """
    Plots a Point marker with set marker, size and color.

    Color can be set to 'random' for random color.
    """
    global fig

    if point is None:
        return

    if isinstance(color, str) and color == "random":
        color = get_random_color()

    if marker == "0":
        marker = "o"
    if text is not None:
        plt.text(point.x, point.y, text, fontsize=10, color=color)
    plt.plot(point.x, point.y, marker=marker, markersize=markersize, color=color)


def show_plot():
    """
    Shows the current plot.

    Blocking function call as long as the plot window is open.
    """
    global fig
    matplotlib.style.use("fast")
    plt.gca().set_aspect("equal")
    plt.gca().set_axis_off()
    plt.show()


def plot_clear():
    """
    Clears the current plot.
    """
    plt.cla()


def plot_update(time = 0.000001):
    """
    Updates the current plot.
    """
    matplotlib.style.use("fast")
    plt.axis("equal")
    plt.axis("off")
    plt.pause(time)


last_hue = 0.0


def get_random_color():
    """
    Returns a random color.

    Random hue and 100% saturation and value in the form of (r, g, b) with values ranging from 0 to 1
    """
    global last_hue
    hue = (last_hue + 0.1 + random.uniform(0.0, 0.05)) % 1.0
    last_hue = hue
    return matplotlib.colors.hsv_to_rgb([hue, 1, 1])


def save_fig(path = ""):
    """
    Saves current plot under the given path.
    """
    plt.axis("off")
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(path, dpi=200)
