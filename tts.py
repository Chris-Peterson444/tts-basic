from typing import Callable
import numpy as np
import sounddevice as sd
from TTS.api import TTS
import pulsectl


def say(
    text: str,
    tts_func: Callable[str, list[int]],
    sample_rate: int = 22050,
):
    wav: list[int] = tts_func(text)

    sd.play(wav, sample_rate, blocking=False)


def main():

    tts = TTS(model_name='tts_models/en/vctk/vits', gpu=True)
    pulse = pulsectl.Pulse('python_tts_client')
    pulse_sink_id = pulse.module_load('module-null-sink', 'sink_name=python_tts')
    pulse_source_id = pulse.module_load('module-virtual-source', 'master=python_tts.monitor source_name=TTSMic')

    # def tts_func(text, tts):
    #     return tts.tts(text, speaker='p230')

    sd._terminate()
    sd._initialize()
    devices = sd.query_devices()

    dev = None
    for d in devices:
        if 'python_tts' in d['name']:
            dev = d
            break



    stream = sd.OutputStream(device=dev['index'], samplerate=48000, channels=1, dtype='float32')
    # stream.start()

    while True:
        text = input('>')
        if text == '!q':
            break
        # if text == 'exit':
        #     break
        # say(text, tts_func, sample_rate=tts.synthesizer.output_sample_rate)
        # wav = tts_func(text)
        wav = tts.tts(text, speaker='p230')

        # print('got wav data')
        data = np.array(wav, dtype='float32')
        data = np.repeat(data, 2)
        # data = np.array([data, data], dtype=np.float32)

        # with stream:
        # stream.start()
        # stream.write(data)
        # stream.stop()
        sd.play(device=dev['index'], data=data)
        # sd.play(data)

    pulse.module_unload(pulse_sink_id)
    pulse.module_unload(pulse_source_id)

    # stream.stop()




if __name__ == '__main__':
    main()
