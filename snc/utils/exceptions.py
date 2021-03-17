def e_assert(condition: bool, exception: Exception):
    """
    Custom assert which raise a specific exception when the condition is not met

    :param condition: the condition that must be met
    :param exception: the exception that should be raised if the condition is not met
    """
    if bool(__debug__):
        if not condition:
            raise exception


class ScipyLinprogStatusError(Exception):
    """Exception raised when the Scipy function linprog returns a status equal to 4"""

    def __init__(self, message: str):
        """
        :param message: explanation of the error
        """
        self.message = message
        super().__init__(self.message)


class ArraySignError(Exception):
    """Exception raised when an array has any or all of its components with different sign
    (i.e. positive or negative) than expected."""

    def __init__(self, array_name: str, all_components: bool, positive: bool,
                 strictly: bool):
        """
        :param array_name: name of the variable assigned to the array which has a wrong sign
        :param all_components: configures if the exception is raised when all the components, or
            when at least one of the components have different sign than expected.
        :param positive: whether the sign should have been positive or negative.
        :param strictly: whether zero components can raise the exception or not (i.e. whether the
            components should be bounded away from zero or zero components are accepted.)
        """
        self.array_name = array_name
        self.all_components = all_components
        self.positive = positive
        self.strictly = strictly

        self.message = self.array_name + " should be "
        if strictly:
            self.message += "strictly "
        if positive:
            self.message += "positive "
        else:
            self.message += "negative "
        if all_components:
            self.message += "for all components."
        else:
            self.message += "for at least one component."

        super().__init__(self.message)


class EmptyArrayError(Exception):
    """Exception raised when an array is empty"""

    def __init__(self, array_name: str):
        """
        :param array_name: the name of the variable assigned to the array being checked.
        """
        self.array_name = array_name
        self.message = self.array_name + " should not be empty."
        super().__init__(self.message)
