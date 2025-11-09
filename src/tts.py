from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
from gemini import retrieve_response

load_dotenv()

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)
speech_file_path = os.path.join(parent_directory, "speech.wav")

while True:
    client = OpenAI()
    inputs = []
    responses = []

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=retrieve_response(inputs, responses),
        instructions="Speak in a cheerful and positive tone.",
        response_format="wav"
    ) as response:
        response.stream_to_file(speech_file_path)

    subprocess.run(["aplay", speech_file_path])