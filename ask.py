import argparse
from build_rag import RAG, CRAG
from utils import get_answer
from audio_utils import get_audio, make_listener, transcribe
import pyttsx3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions")
    parser.add_argument("--transcribe", action="store_true", help="Use speech-to-text via microphone input")
    parser.add_argument("--CRAG", action="store_true", help="Use CRAG")
    args = parser.parse_args()

    if args.transcribe: # use a microphone to ask questions
        # speech-to-text
        print("Using Speech-to-Text")
        model = make_listener("base")
        audio_dir="audio/audio.wav"
        # text-to-speech (TTS)
        speaker = pyttsx3.init()
        voiceRate = 160 # words per minute
        speaker.setProperty('rate',voiceRate)
        voice = speaker.getProperty('voices') # voice
        speaker.setProperty('voice', voice[2].id)

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
                speaker.say(response)
                speaker.runAndWait()
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
                speaker.say(response)
                speaker.runAndWait()
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
