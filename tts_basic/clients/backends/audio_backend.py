""" Describes the AudioBackend ABC for audio backends to implement. """
from abc import ABC
from abc import abstractmethod
from typing import Optional


class AudioBackend(ABC):
    """ An abstract class for an audio backend. """

    @abstractmethod
    def __init__(
            self,
            source_device,
            sample_rate: Optional[int],
            channels: Optional[int],
            dtype: Optional[str],
    ) -> None:
        pass

    @abstractmethod
    def play(
        self,
        data: list[list[float]],
        blocking: Optional[bool] = False,
    ) -> None:
        pass
