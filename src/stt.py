import subprocess
import time

from dotenv import load_dotenv
from openai import OpenAI

# Use the same device & params that worked for you
ARECORD_CMD = [
    "arecord",
    "-D", "plughw:3,0",
    "-f", "S16_LE",
    "-r", "16000",
    "-c", "1",
    "-d", "5",
    "tmp.wav",
]

load_dotenv()
client = OpenAI()


def record_clip():
    print("Recording 5 s...")
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
