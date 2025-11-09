import subprocess
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)
input_file_path = os.path.join(parent_directory, "tmp.wav")

ARECORD_CMD = [
    "arecord",
    "-D", "plughw:1,0",
    "-f", "S16_LE",
    "-r", "16000",
    "-c", "1",
    "-d", "10",
    input_file_path,
]

load_dotenv()
client = OpenAI()


def record_clip():
    print("Recording 10 s...")
    subprocess.run(ARECORD_CMD, check=True)
    print("Recording done.")


def transcribe_file(path: str) -> str:
    with open(path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )
    return transcription.text.strip()


def get_transcript(log_latency: bool = True) -> str:
    """
    High-level function for the rest of the team:
    record audio, return transcript string and latency.
    """
    record_clip()
    start = time.perf_counter()
    text = transcribe_file("tmp.wav")
    latency = time.perf_counter() - start
    if log_latency:
        print(f"Transcription latency: {latency:.2f} s")
    return text


if __name__ == "__main__":
    input("Press Enter, then talk for 5 seconds...")
    txt = get_transcript()
    print("Transcript:", repr(txt))
