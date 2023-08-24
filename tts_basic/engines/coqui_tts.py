from typing import Optional

from .tts_engine import TTSEngine

from TTS.api import TTS


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
