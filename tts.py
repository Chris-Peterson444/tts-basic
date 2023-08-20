import sys
from abc import ABC
from abc import abstractmethod
from types import TracebackType
from typing import Callable
from typing import Optional
from typing import Type
from typing import Union
from typing import Generator
from queue import Queue

import numpy as np
import pulsectl
import sounddevice as sd
from TTS.api import TTS
import jack


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


class CoquiTTS(TTSEngine):
    def __init__(
        self,
        model_name: str,
        speaker_name: Optional[str] = None,
        language: Optional[str] = None,
        gpu: Optional[bool] = False,
    ):
        self.tts = TTS(model_name=model_name, gpu=gpu)
        self.speaker_name = speaker_name
        self.language = language

    def say(self, text: str) -> list[list[float]]:
        wav: list[float] = self.tts.tts(text, speaker=self.speaker_name)
        return [wav]

    @property
    def sample_rate(self):
        return self.tts.synthesizer.output_sample_rate


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


class PortAudioBackend(AudioBackend):
    """ An audio backend wrapper class for PortAudio via sounddevice. """

    def __init__(
            self,
            source_device: Optional[Union[int, str]] = None,
            sample_rate: Optional[int] = 48000,
            channels: Optional[int] = 2,
            dtype: Optional[str] = 'float32',
    ) -> None:

        # Purposely only outputing to one device
        self.source_device = source_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

        self._device_obj: dict = self._select_device(self.source_device)
        self._device_index = self._device_obj['index']
        # print(f'Using device {self._device_obj!r}')

    def _select_device(
            self,
            device: Optional[Union[int, str]],
    ) -> dict:
        """ Select the audio device. """

        if device is None:
            return None
        if isinstance(device, int):
            return device
        elif isinstance(device, str):
            devices = self.query_devices()
            for dev in devices:
                if device in dev['name']:
                    return dev

            raise ValueError(f'Could not find device with name {device!r}')
        else:
            raise TypeError('device must be an int or str')

    def query_devices(self) -> sd.DeviceList:
        """ Query the audio devices. """

        # This is a hack to get around a bug in PortAudio/sounddevice (devices
        # don't get repopulated unless the library is reloaded)
        sd._terminate()
        sd._initialize()

        return sd.query_devices()

    def play(
        self,
        data: np.ndarray,
        blocking: Optional[bool] = False,
    ) -> None:
        """ Play the audio data. """

        channels, samples = data.shape
        if channels == 1:
            data = data[0]
        try:
            sd.play(
                data=data,
                device=self._device_index,
                samplerate=self.sample_rate,
                blocking=blocking
            )
        except Exception:
            print(
                f'Play settings: device={self._device_obj},',
                f'samplerate={self.sample_rate}, blocking={blocking}',
                file=sys.stderr,
            )
            raise


class AudioServer(ABC):
    """ An abstract class for an audio server.

    This class is used to create a context manager to handle creating and
    deleting audio devices (sources or sinks). This can optionally be a full
    class to handle Audio Server state and API calls, but at minimum must
    handle creating and deleting audio devices.

    Any devices created by the audio server should be done in __enter__
    and should be destroyed in __exit__. This ensures that the devices
    do not persist after the program exits. If you are setting up the audio
    devices separately, you probably don't need to use this class. Instead,
    you can just use an audio backend wrapper class to create audio stream and
    pipe the output to the existing audio device.
    """

    @ abstractmethod
    def __enter__(self) -> 'AudioServer':
        pass

    @ abstractmethod
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


class PulseAudioServer(AudioServer):
    """ A PulseAudio server wrapper class via `pulsectl`.

    Also works with PipeWire PulseAudio plugin. Since `pulsectl` has yet to
    support submitting audio data to a source, this class also relies on an
    audio backend wrapper class to create an audio stream and pipe the output
    to the source.
    """

    DUMMY_SINK_NAME = 'tts_dummy_sink'  # dummy sink for source master

    def __init__(
        self,
        client_name: Optional[str] = 'TTSBasiClient',
        sink_name: Optional[str] = 'TTSBasicSink',
        source_name: Optional[str] = 'TTSBasicSource',
        sample_rate: Optional[int] = 48000,
        channels: Optional[int] = 2,
        dtype: Optional[str] = 'float32',
        audio_backend: Optional[Type[AudioBackend]] = PortAudioBackend,
        auto_resample: Optional[bool] = True,
    ) -> None:

        self.pulse = pulsectl.Pulse(client_name)
        self.sink_name: str = sink_name
        self.source_name: str = source_name
        self._sink_id: Optional[str] = None
        self._source_id: Optional[str] = None

        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.audio_backend_constructor: AudioBackend = audio_backend

        self.auto_resample = auto_resample

        self.dummy_sink_index: Optional[str] = None

    def _load_module(
            self,
            modle_name: str,
            args: Optional[str] = None,
    ) -> str:
        """ Load a PulseAudio module. """

        return self.pulse.module_load(name=modle_name, args=args)

    def _unload_module(
            self,
            module_id: str,
    ) -> None:
        """ Unload a PulseAudio module. """

        self.pulse.module_unload(module_id)

    def _create_sinks(self) -> None:
        """ Create sinks on the audio server. """

        args = (
            f'sink_name={self.sink_name} '
            f'rate={self.sample_rate} '
            f'channels={self.channels} '
            f'format={self.dtype}'
        )

        self._sink_id = self._load_module(
            'module-null-sink',
            args=args,
        )

    def _create_sources(self, master: str) -> None:
        """ Create sources on the audio server. """

        args = (
            f'source_name={self.source_name} '
            # f'rate={self.sample_rate} '
            f'channels={self.channels} '
            f'format={self.dtype} '
            f'master={master} '
        )

        self._source_id = self._load_module(
            'module-virtual-source',
            args=args,
        )

    def _create_dummy_sink(self) -> None:
        """ Create a dummy sink on the audio server. """

        self.dummy_sink_index = self._load_module(
            'module-null-sink',
            args=f'sink_name={self.DUMMY_SINK_NAME}',
        )

    def __enter__(self) -> AudioServer:
        self._create_dummy_sink()
        self._create_sources(master=f'{self.DUMMY_SINK_NAME}.monitor')

        self._audio_backend_instance = self.audio_backend_constructor(
            source_device=self.DUMMY_SINK_NAME,
            # source_device=self.source_name,
            # sample_rate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
        )

        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType],
    ) -> Union[bool, None]:
        # self._unload_module(self._sink_id)
        self._unload_module(self._source_id)
        self._unload_module(self.dummy_sink_index)

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

        self._audio_backend_instance.play(data, blocking=blocking)


class JackAudioServer(AudioServer):
    ''' Audio server using the JACK audio server via JACK-Client.

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

    def __enter__(self) -> AudioServer:
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


def say(
    text: str,
    tts_func: Callable[str, list[int]],
    sample_rate: int = 22050,
):
    wav: list[int] = tts_func(text)

    sd.play(wav, sample_rate, blocking=False)


def main() -> int:

    tts_engine = CoquiTTS(
        model_name='tts_models/en/vctk/vits',
        gpu=True,
        speaker_name='p230',
    )
    # pulse_client = PulseAudioServer(sample_rate=44100)
    #
    # with pulse_client as pulse:
    #     while True:
    #         text = input('>')
    #         if text == '!q':
    #             break
    #
    #         data = tts_engine.say(text)
    #         pulse.play(
    #             data=data,
    #             samplerate=tts_engine.sample_rate,
    #             blocking=False,
    #         )

    with JackAudioServer() as jack:
        while True:
            text = input('>')
            if text == '!q':
                break

            data = tts_engine.say(text)
            jack.play(
                data=data,
                samplerate=tts_engine.sample_rate,
                blocking=False,
            )


if __name__ == '__main__':
    main()
