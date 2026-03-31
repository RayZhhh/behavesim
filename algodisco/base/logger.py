# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

from abc import abstractmethod, ABC
from typing import Dict, Any


class AlgoSearchLoggerBase(ABC):

    def set_log_item_flush_frequency(self, *args, **kwargs): ...

    @abstractmethod
    def log_dict(self, log_item: Dict, item_name: str):
        """Synchronous version of log_dict. Override in subclass if needed."""
        raise NotImplementedError("Sync version not implemented for this logger")

    def finish(self):
        """Synchronous version of finish. Override in subclass if needed."""
        pass
