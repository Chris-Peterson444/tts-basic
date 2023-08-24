import sys
from typing import Optional
from typing import Union

import numpy as np
import sounddevice as sd

from .audio_backend import AudioBackend


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
