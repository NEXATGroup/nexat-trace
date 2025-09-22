

class RoutePlanningError(Exception):
    """
    Exception that indicates an error in route planning.
    """

    def __init__(self, msg):
        super().__init__(msg)


class GraphConstructionError(Exception):
    """
    Exception that indicates an error on Graph Construction.
    """

    def __init__(self, msg):
        super().__init__(msg)


class GraphConstraintViolationError(Exception):
    """
    Exception that indicates a violation of Field graph constraints.
    """

    def __init__(self, msg):
        super().__init__(msg)


class PlannerConfigurationError(Exception):
    """
    Exception that indicates an error in the planner configuration.
    """

    def __init__(self, msg):
        super().__init__(msg)
