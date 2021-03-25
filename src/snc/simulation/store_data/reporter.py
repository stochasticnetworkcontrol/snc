from collections import defaultdict
from typing import List, Dict, Any, Optional

import snc.utils.snc_types as types
from snc.simulation.plot.base_handlers import Handler


class Reporter:
    """
    A class to report on live experiments.

    This class acts as both a storage object for artifacts produced mid-simulation/mid-calculation,
    and also as a coordinator of 'reports' of these artifacts. These 'reports' are generated
    by Handlers, registered with the Reporter at initialisation, and set off by the Reporter.

    This architecture allows the data collected and stored _once_ by the reporter to generate
    multiple different useful outputs - whether that is simply printing values to std out, or
    actually updating a live bar plot, or surfaces plot.
    """

    def __init__(self, handlers: Optional[List[Handler]] = None):
        self._cache = defaultdict(list)  # type: Dict[str, Any]
        self._handlers = handlers if handlers is not None else []

    @property
    def handlers(self):
        """The handlers registered with the reporter"""
        return self._handlers

    @property
    def cache(self):
        """The data store of the object"""
        return self._cache

    def store(self, **kwargs):
        """Store multiple items in the store"""
        for k, v in kwargs.items():
            self.cache[k].append(v)

    def report(self, data_dict: types.DataDict, step: int):
        """Generate all the reports at this time step, by calling all the handlers.

        :param data_dict: The current data dictionary at this time step.
        :param step: The current step of the simulation.
        """
        for handler in self._handlers:
            handler(self.cache, data_dict,  step)
