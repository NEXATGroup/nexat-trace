from nexat_trace.shared.weights import Weights


class EdgeMetrics:
    """
    Class that represents edge traversal data.

    Holds information about
    - distances
    - driving time
    - wether or not the edge is a previously driven track
    - curve angle
    - cost offset

    Can represent an ab lines as well as some other connection on a track graph.
    """

    def __init__(self):
        # measurements
        self.distance = 0.0  # m
        self.time = 0.0  # s
        self.angle = 0.0  # pi

        # further cost calculation parameters
        self.cost_offset = 0.0
        self.is_ab = False
        self.working_corridor_error = 0.0  # [0.0, ~1.0]
        self.is_neighbor_curve = False
        self.missed_path_share = 0.0

    def add(self, other):
        """
        Adds 2 metrics together and returns a new instance.
        """
        if self is None or other is None:
            if self is None:
                return other
            if other is None:
                return self
            return EdgeMetrics()

        metrics = EdgeMetrics()
        metrics.distance = self.distance + other.distance
        metrics.time = self.time + other.time

        if self.angle > other.angle:
            metrics.angle = self.angle
        else:
            metrics.angle = other.angle

        metrics.cost_offset = self.cost_offset + other.cost_offset
        metrics.is_ab = self.is_ab and other.is_ab
        metrics.is_neighbor_curve = self.is_neighbor_curve and other.is_neighbor_curve
        metrics.working_corridor_error = self.working_corridor_error + other.working_corridor_error
        metrics.missed_path_share = self.missed_path_share + other.missed_path_share

        return metrics

    def copy(self):
        """
        Returns a deep copy of the instance.
        """
        cpy = EdgeMetrics()
        cpy.distance = self.distance
        cpy.angle = self.angle
        cpy.cost_offset = self.cost_offset
        cpy.time = self.time
        cpy.is_ab = self.is_ab
        cpy.working_corridor_error = self.working_corridor_error
        cpy.is_neighbor_curve = self.is_neighbor_curve
        cpy.missed_path_share = self.missed_path_share

        return cpy

    def __str__(self) -> str:
        """
        Returns string representation of instance.
        """

        if self.distance > 1000.0:
            distance_str = f"{self.distance / 1000.0:.2f} km"
        else:
            distance_str = f"{self.distance:.2f} m"

        if self.time < 60.0:
            time_str = f"{self.time} seconds"
        elif self.time > 60.0 and self.time < 3600:
            time_str = f"{self.time / 60:.1f} minutes"
        elif self.time > 3600:
            time_str = f"{self.time / 3600:.1f} hours"

        avg_speed_str = "Avg. Speed:     -\n"

        if self.time > 0.0:
            avg_speed_str = (
                f"Avg. Speed:     {(self.distance) / (self.time) * 3.6:.2f} km/h\n"
            )

        return (
            f"Distance:       {distance_str}\n"
            + f"Time:           {time_str}\n"
            + avg_speed_str
        )

    def get_cost(self, weights: Weights):
        """
        Calculates the cost with given weights.
        """
        cost = 0.0

        if self.is_ab:
            return cost

        turn_cost = 0.0
        if self.angle > 0.0:
            angle_penalty = ((self.angle * 2.0) ** weights.turn_cost_angle_exponent)
            turn_cost += angle_penalty
            turn_cost *= weights.turn_cost_gain
            turn_cost = turn_cost ** weights.turn_cost_exponent

        distance_cost = self.distance * weights.headland_distance_factor
        distance_cost = distance_cost + self.missed_path_share * weights.missed_path_penalty
        distance_cost = distance_cost ** weights.headland_cost_exponent
        cost += distance_cost

        cost += turn_cost
        cost += self.cost_offset

        cost += weights.corridor_error_cost * self.working_corridor_error

        cost *= weights.global_cost_gain
        cost += weights.global_cost_offset

        return cost
