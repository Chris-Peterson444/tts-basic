from typing import Optional
from typing import Type
from typing import Union
from typing import Callable
from typing import Generator
from types import TracebackType

from queue import Queue

import jack
import numpy as np

from .audio_client import AudioClient


class JackAudioClient(AudioClient):
    ''' Audio client using the JACK audio server via JACK-Client.

    This class doesn't require a separate audio backend, as the JACK api
    handles the audio output itself. This will create a JACK Port on the
    audio graph, but may not be visible as a device in your application's
    GUI. '''

    def __init__(
            self,
            client_name: Optional[str] = 'TTSBasiClient',
            output_port_name: Optional[str] = 'TTSBasicOutput',
            auto_resample: Optional[bool] = True,
            dtype: Optional[str] = 'float32',
    ) -> None:

        self.client = jack.Client(client_name)
        self.output_port_name = output_port_name
        self.auto_resample = auto_resample
        self.dtype = dtype

        self.sample_rate = self.client.samplerate
        self.block_size = self.client.blocksize

        self.client.set_samplerate_callback(self._update_samplerate())
        self.client.set_process_callback(self._process_callback())
        self.client.set_blocksize_callback(self._update_blocksize())

        self.data_queue: Queue[np.ndarray] = Queue()

    def _update_samplerate(self) -> Callable[[int], None]:

        def callback(sample_rate: int) -> None:
            self.sample_rate = sample_rate

        return callback

    def _update_blocksize(self) -> Callable[[int], None]:

        def callback(block_size: int) -> None:
            self.block_size = block_size

        return callback

    def _process_callback(self) -> Callable[[int], None]:

        def callback(frames: int) -> None:
            # Don't stop playback on empty queue, just play silence
            if self.data_queue.empty():
                return None

            # Get data from queue and write to JACK output ports
            data = self.data_queue.get_nowait()
            for channel, port in zip(data, self.client.outports):
                # print(port.get_array().shape)
                port.get_array()[:] = channel

        return callback

    def __enter__(self) -> AudioClient:
        self.client.activate()
        _ = self.client.outports.register(self.output_port_name)
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType],
    ) -> Union[bool, None]:
        # self.client.inports.clear()  # unregister all inports
        # self.client.outports.clear()  # unregister all outports
        self.client.deactivate()
        self.client.close()

    def resample(
            self,
            data: list[list[float]],
            original_rate: int,
    ) -> np.ndarray:
        """ Resample the audio data to the sample rate of the audio server. """

        data = np.array(data, dtype=self.dtype)
        # TODO: Do real resampling
        # Currently works for CoquiTTS models, which return 22050 Hz audio
        # This means it always samples up to 44100 Hz, but it sounds
        # acceptable for now
        data = np.repeat(data, self.sample_rate//original_rate, axis=1)
        return data

    def _split_blocks(
            self,
            data: np.ndarray,
    ) -> Generator[np.ndarray, None, None]:
        """ Split data into blocks of size self.block_size. """
        block_count = data.shape[1] // self.block_size
        for i in range(block_count):
            yield data[:, i*self.block_size:(i+1)*self.block_size]

    def play(
            self,
            data: Union[list[list[float]], np.ndarray],
            blocking: Optional[bool] = False,
            samplerate: Optional[int] = None,
    ) -> None:
        """ Play audio data on the audio server. """

        if isinstance(data, list):
            data = np.array(data, dtype=self.dtype)

        if self.auto_resample and samplerate != self.sample_rate:
            data = self.resample(data=data, original_rate=samplerate)

        for block in self._split_blocks(data):
            self.data_queue.put_nowait(block)
