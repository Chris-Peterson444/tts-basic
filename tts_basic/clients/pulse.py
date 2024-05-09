from types import TracebackType
from typing import Optional, Type, Union

import numpy as np
import pulsectl

from .audio_client import AudioClient
from .backends.audio_backend import AudioBackend
from .backends.portaudio import PortAudioBackend


class PulseAudioClient(AudioClient):
    """A class to interact a PulseAudio server via `pulsectl`.

    Also works with PipeWire PulseAudio plugin. Since `pulsectl` has yet to
    support submitting audio data to a source, this class also relies on an
    audio backend wrapper class to create an audio stream and pipe the output
    to the source.
    """

    DUMMY_SINK_NAME = "tts_dummy_sink"  # dummy sink for source master

    def __init__(
        self,
        client_name: str = "TTSBasiClient",
        sink_name: str = "TTSBasicSink",
        source_name: str = "TTSBasicSource",
        sample_rate: int = 48000,
        channels: int = 2,
        dtype: str = "float32",
        audio_backend: Type[AudioBackend] = PortAudioBackend,
        auto_resample: bool = True,
    ) -> None:

        self.pulse = pulsectl.Pulse(client_name)
        self.sink_name: str = sink_name
        self.source_name: str = source_name
        self._sink_id: str = ""
        self._source_id: str = ""

        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.audio_backend_constructor: Type[AudioBackend] = audio_backend

        self.auto_resample = auto_resample

        self.dummy_sink_index: str = ""

    def _load_module(
        self,
        modle_name: str,
        args: Optional[str] = None,
    ) -> str:
        """Load a PulseAudio module."""

        print(f"{args=}")
        return self.pulse.module_load(name=modle_name, args=args)

    def _unload_module(
        self,
        module_id: str,
    ) -> None:
        """Unload a PulseAudio module."""

        self.pulse.module_unload(module_id)

    def _create_sinks(self) -> None:
        """Create sinks on the audio server."""

        args = (
            f"sink_name={self.sink_name} "
            f"rate={self.sample_rate} "
            f"channels={self.channels} "
            f"format={self.dtype}"
        )

        self._sink_id = self._load_module(
            "module-null-sink",
            args=args,
        )

    def _create_source(self, master: str) -> None:
        """Create sources on the audio server."""

        args = (
            f"source_name={self.source_name} "
            # f'rate={self.sample_rate} '
            f"channels={self.channels} "
            f"format={self.dtype} "
            f"master={master} "
        )

        print(f"{args=}")
        self._source_id = self._load_module(
            # "module-virtual-source",
            "module-remap-source",
            args=args,
        )

    def _load_dummy_sink(self) -> None:
        """Create a dummy sink on the audio server."""

        self.dummy_sink_index = self._load_module(
            "module-null-sink",
            args=f"sink_name={self.DUMMY_SINK_NAME}",
        )

    def _unload_dummy_sink(self) -> None:
        """Unload the dummy sink on the audio server."""

        if self.dummy_sink_index != "":
            self._unload_module(self.dummy_sink_index)
            self.dummy_sink_index = ""

    def __enter__(self) -> AudioClient:
        self._load_dummy_sink()
        self._create_source(master=f"{self.DUMMY_SINK_NAME}.monitor")

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
        self._unload_dummy_sink()

        return None

    def resample(
        self,
        data: Union[list[list[float]], np.ndarray],
        original_rate: int,
    ) -> np.ndarray:
        """Resample the audio data to the sample rate of the audio server."""

        if isinstance(data, list):
            np_data: np.ndarray = np.array(data, dtype=self.dtype)
        else:
            np_data: np.ndarray = data  # type: ignore

        # TODO: Do real resampling
        # Currently works for CoquiTTS models, which return 22050 Hz audio
        # This means it always samples up to 44100 Hz, but it sounds
        # acceptable for now
        np_data = np.repeat(np_data, self.sample_rate // original_rate, axis=1)
        return np_data

    def play(
        self,
        data: Union[list[list[float]], np.ndarray],
        blocking: Optional[bool] = False,
        samplerate: Optional[int] = None,
    ) -> None:
        """Play audio data on the audio server."""

        if isinstance(data, list):
            np_data: np.ndarray = np.array(data, dtype=self.dtype)
        else:
            np_data: np.ndarray = data  # type: ignore

        if samplerate is None:
            samplerate = self.sample_rate

        if self.auto_resample and samplerate != self.sample_rate:
            np_data = self.resample(data=np_data, original_rate=samplerate)

        self._audio_backend_instance.play(np_data, blocking=blocking)
