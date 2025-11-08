from google import genai
from dotenv import load_dotenv
import time

load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain how AI works in a few words",
)

print(response.text)