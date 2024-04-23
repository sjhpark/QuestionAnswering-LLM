import argparse
from build_rag import RAG, CRAG
from utils import get_answer
from audio_utils import get_audio, make_listener, transcribe
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.api import TTS
import torch
import pyaudio
import wave

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions")
    parser.add_argument("--transcribe", action="store_true", help="Use speech-to-text via microphone input")
    parser.add_argument("--CRAG", action="store_true", help="Use CRAG")
    args = parser.parse_args()

    if args.transcribe: # use a microphone to ask questions
        # speech-to-text model initialization
        print("Using Speech-to-Text")
        model = make_listener("base")
        audio_dir="audio/audio.wav"

        # text-to-speech (TTS) model initialization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Init TTS with the target model name
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)

        if args.CRAG:
            from utils_CRAG import get_censored_answer
            print("Using CRAG")
            app = CRAG()
            while True:
                input("Press Enter to ask a question using your microphone...")
                # record audio
                get_audio(audio_dir, record_duration=8)
                # transcribe audio to text
                transcript = transcribe(model, audio_dir)
                response = get_censored_answer(app, transcript)
                response = response["keys"]["generation"]
                # speak
                tts.tts_to_file(text=response, file_path="audio/output.wav") # convert text to wav file
                p = pyaudio.PyAudio() # initialize PyAudio
                wf = wave.open("audio/output.wav", 'rb')
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)
                data = wf.readframes(1024)
                # play stream
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
                # stop stream
                stream.stop_stream()
                stream.close()
                p.terminate()
        else:
            print("Using RAG")
            qa_chain = RAG()
            while True:
                input("Press Enter to ask a question using your microphone...")
                # record audio
                get_audio(audio_dir, record_duration=8)
                # transcribe audio to text
                transcript = transcribe(model, audio_dir)
                response = get_answer(qa_chain, transcript)
                response = response["result"]
                # speak
                tts.tts_to_file(text=response, file_path="audio/output.wav") # convert text to wav file
                p = pyaudio.PyAudio() # initialize PyAudio
                wf = wave.open("audio/output.wav", 'rb')
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)
                data = wf.readframes(1024)
                # play stream
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
                # stop stream
                stream.stop_stream()
                stream.close()
                p.terminate()
    else:
        if args.CRAG:
            from utils_CRAG import get_censored_answer
            print("Using CRAG")
            app = CRAG()
            while True:
                query = input(f"Type in your question:")
                get_censored_answer(app, query)
        else:
            print("Using RAG")
            qa_chain = RAG()
            while True:
                query = input(f"Type in your question:")
                get_answer(qa_chain, query)
