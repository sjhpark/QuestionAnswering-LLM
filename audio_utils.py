import argparse
import whisper
import pyaudio
import wave
from utils import color_print

def make_listener(size:str):
    """
    OpenAI Whisper (https://github.com/openai/whisper)
    Size 	Parameters 	English-only model 	Multilingual model 	Required VRAM 	Relative speed
    tiny 	39 M 	    tiny.en 	        tiny 	            ~1 GB 	        ~32x
    base 	74 M 	    base.en 	        base 	            ~1 GB 	        ~16x
    small 	244 M 	    small.en 	        small 	            ~2 GB 	        ~6x
    medium 	769 M 	    medium.en 	        medium 	            ~5 GB 	        ~2x
    large 	1550 M 	    N/A 	            large 	            ~10 GB 	        1x
    """
    return whisper.load_model(size)

def get_audio(audio_dir:str="audio/audio.wav", record_duration:int=8):
    # record audio from your microphone and save it to a wav file
    # code reference: https://stackoverflow.com/questions/35344649/reading-input-sound-signal-using-python
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    color_print("* Recording... please speak into the microphone", 'yellow')

    frames = []
    for i in range(0, int(RATE / CHUNK * record_duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(audio_dir, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe(model, audio_dir):
    speech = model.transcribe(audio_dir)
    transcript = speech["text"]
    color_print(f'Transcribed Speech: \n {transcript}', 'yellow')
    return transcript