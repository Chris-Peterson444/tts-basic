""" Describes the AudioClient ABC for audio clients to implement. """

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Optional, Type, Union


class AudioClient(ABC):
    """An abstract class for an audio client to interface with the system
    audio server (e.g., pulse, jack, pipewire).

    This class is used to create a context manager to handle creating and
    deleting audio devices (sources or sinks). This can optionally be a full
    class to handle server state and wrap various API calls, but at minimum
    must handle creating and deleting audio devices.

    Any devices created by the audio client should be done in __enter__
    and should be destroyed in __exit__. This ensures that the devices
    do not persist after the program exits. If you are setting up the audio
    devices separately, you probably don't need to use this class. Instead,
    you can just use an audio backend wrapper class to create audio stream and
    pipe the output to the existing audio device.
    """

    @abstractmethod
    def __enter__(self) -> "AudioClient":
        pass

    @abstractmethod
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Union[bool, None]:
        pass

    # @abstractmethod
    # def play(
    #     self,
    #     data: list[list[float]],
    #     blocking: Optional[bool] = False,
    # ) -> None:
    #     pass
