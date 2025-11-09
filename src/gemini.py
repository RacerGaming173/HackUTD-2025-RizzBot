from google import genai
from dotenv import load_dotenv
from prompt_template import SYSTEM_PROMPT
from stt import get_transcript
import time

load_dotenv()

def retrieve_response(prior_inputs, prior_responses):

    client = genai.Client()

    input_conversation = get_transcript()

    string_prior_inputs = ",\n".join(prior_inputs)
    string_prior_responses = ",\n".join(prior_responses)
    prompt = SYSTEM_PROMPT + "\nPrior inputs: " + string_prior_inputs + "\nPrior responses: " + string_prior_responses  + "\nInput sentence: " + input_conversation

    start = time.perf_counter()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    latency = time.perf_counter() - start
    print("Gemini response time: " + str(latency))
    
    response_string = response.text
    response_string = response.text[len('"AI Suggestion:'):] if "AI Suggestion: " in response.text else response.text
    response_string = response_string.replace('"', "")

    prior_inputs.append(input_conversation)
    prior_responses.append(response_string)
    print(response_string)

if __name__ == "__main__":
    previous_input_conversations = []
    previous_responses = []

    while True:
        retrieve_response(previous_input_conversations, previous_responses)
    