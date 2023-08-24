""" Describes the TTSEngine ABC class that all TTS engines should implement. """

from abc import ABC
from abc import abstractmethod


class TTSEngine(ABC):
    """ An abstract class for a TTS engine.

    This class is used to simplfy interfacing with different TTS engines.
    """

    @abstractmethod
    def say(self, text: str) -> list[list[float]]:
        """ The `say` method should at least take a string and return the audio
        data in a list of lists of floats. The size of the outer list is the
        number of channels in the audio data, with the size of the inner lists
        being the number of samples. If the TTS engine supports multiple
        speakers or languages, you can add those as optional parameters to the
        `say` method.
        """
        pass
