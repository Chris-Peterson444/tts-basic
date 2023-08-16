import sounddevice as sd
import soundfile as sf
import numpy as np

# Extract data and sampling rate from File
data, sr = sf.read('test.wav')

# set type and upsample the data
data = np.array(data, dtype='float32')
data = np.repeat(data, 2)


# setup stream object
tts_stream = sd.OutputStream(samplerate=48000, channels=1, dtype='float32')

# send data to tts_stream
with tts_stream as s:
    s.write(data)

