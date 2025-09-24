
class Weights:
    """
    Class to configure weight profiles for different optimization strategies.

    Turn parameters
    ---------------

    turn_cost_gain: turn cost multiplier
    turn_cost_exponent: turn cost = turn cost ^ turn_cost_exponent
    turn_angle_cost_exponent: cost = target angle difference ^ turn_angle_cost_exponent
    turn_cost_angle_exponent: penalizes edge cost if turn angle is > 90°

    higher turn cost value => prioritize more favorable turns

    Distance parameters
    -------------------

    headland_distance_factor: headland cost = distance * headland_distance_factor
    headland_cost_exponent: headland cost = cost ^ headland_cost_exponent

    higher headland cost value => prioritize shorter overall distance

    Route parameters
    ----------------

    corridor_error_cost: cost += corridor_error_cost if detected,
    missed_path_penalty: distance_cost = distance_cost + self.missed_path_share * weights.missed_path_penalty
    different_block_penalty: cost += different_block_penalty if calculating cost between different blocks

    Global parameters
    -----------------

    global_cost_gain: cost = cost * global_cost_gain
    global_cost_offset: cost = cost + global_cost_offset
    """

    ONLY_DISTANCE = 0
    GRAPH_BUILDING = 1

    @staticmethod
    def from_input(
            turn_cost_gain = 2.0,
            turn_cost_exponent = 0,
            turn_cost_angle_exponent = 0,
            headland_distance_factor = 1.5,
            headland_cost_exponent = 1,
            corridor_error_cost = 200,
            missed_path_penalty = 1000.0,
            different_block_penalty = 500.0,
            global_cost_offset = 0.0,
            global_cost_gain = 1.0):
        """
        Returns a new Weights instance constructed from params.
        """
        weights = Weights()

        # turn cost weights
        weights.turn_cost_gain = float(turn_cost_gain)
        weights.turn_cost_exponent = int(turn_cost_exponent)
        weights.turn_cost_angle_exponent = int(turn_cost_angle_exponent)

        # headland cost weights
        weights.headland_distance_factor = float(headland_distance_factor)
        weights.headland_cost_exponent = int(headland_cost_exponent)

        # route weights
        weights.corridor_error_cost = float(corridor_error_cost)
        weights.missed_path_penalty = float(missed_path_penalty)
        weights.different_block_penalty = float(different_block_penalty)

        # global weights
        weights.global_cost_offset = float(global_cost_offset)
        weights.global_cost_gain = float(global_cost_gain)

        return weights

    def __init__(self, profile = ONLY_DISTANCE):

        if profile == Weights.ONLY_DISTANCE:
            # turn cost weights
            self.turn_cost_gain = 0.0
            self.turn_cost_exponent = 0
            self.turn_cost_angle_exponent = 0

            # headland cost weights
            self.headland_distance_factor = 1.0
            self.headland_cost_exponent = 1

            # route weights
            self.corridor_error_cost = 0.0
            self.missed_path_penalty = 1000
            self.different_block_penalty = 0

            # global weights
            self.global_cost_offset = 0.0
            self.global_cost_gain = 1.0

        elif profile == Weights.GRAPH_BUILDING:
            # turn cost weights
            self.turn_cost_gain = 1.0
            self.turn_cost_exponent = 1
            self.turn_cost_angle_exponent = 0

            # headland cost weights
            self.headland_distance_factor = 1
            self.headland_cost_exponent = 1

            # route weights
            self.corridor_error_cost = 0.0
            self.missed_path_penalty = 1000
            self.different_block_penalty = 0

            # global weights
            self.global_cost_offset = 0.0
            self.global_cost_gain = 1.0

    def copy(self):
        """
        Returns a deep copy of the instance.
        """
        new = Weights()
        new.turn_cost_gain = self.turn_cost_gain
        new.turn_cost_exponent = self.turn_cost_exponent
        new.turn_cost_angle_exponent = self.turn_cost_angle_exponent
        new.headland_cost_exponent = self.headland_cost_exponent
        new.headland_distance_factor = self.headland_distance_factor
        new.corridor_error_cost = self.corridor_error_cost
        new.missed_path_penalty = self.missed_path_penalty
        new.different_block_penalty = self.different_block_penalty
        new.global_cost_offset = self.global_cost_offset
        new.global_cost_gain = self.global_cost_gain
        return new

    def __str__(self) -> str:
        """
        Returns string representation of instance.
        """
        ret = "Weights:\n"
        ret += f"turn_cost_gain: {self.turn_cost_gain}\n"
        ret += f"turn_cost_exponent: {self.turn_cost_exponent}\n"
        ret += f"turn_cost_angle_exponent: {self.turn_cost_angle_exponent}\n"
        ret += f"headland_distance_factor: {self.headland_distance_factor}\n"
        ret += f"headland_cost_exponent: {self.headland_cost_exponent}\n"
        ret += f"corridor_error_cost: {self.corridor_error_cost}\n"
        ret += f"missed_path_penalty: {self.missed_path_penalty}\n"
        ret += f"different_block_penalty: {self.different_block_penalty}\n"
        ret += f"global_cost_offset: {self.global_cost_offset}\n"
        ret += f"global_cost_gain: {self.global_cost_gain}\n"
        return ret
