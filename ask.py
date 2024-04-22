import argparse
from build_rag import RAG, CRAG
from utils import get_answer
from audio_utils import get_audio, make_listner, transcribe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions")
    parser.add_argument("--transcribe", action="store_true", help="Use speech-to-text via microphone input")
    parser.add_argument("--CRAG", action="store_true", help="Use CRAG")
    args = parser.parse_args()

    if args.transcribe: # use a microphone to ask questions
        print("Using Speech-to-Text")
        model = make_listner("base")
        audio_dir="audio/audio.wav"

        if args.CRAG:
            from utils_CRAG import get_censored_answer
            print("Using CRAG")
            app = CRAG()
            while True:
                input("Press Enter to ask a question using your microphone...")
                get_audio(audio_dir)
                transcript = transcribe(model, audio_dir)
                get_censored_answer(app, transcript)
        else:
            print("Using RAG")
            qa_chain = RAG()
            while True:
                input("Press Enter to ask a question using your microphone...")
                get_audio(audio_dir)
                transcript = transcribe(model, audio_dir)
                get_answer(qa_chain, transcript)

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
