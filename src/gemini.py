from google import genai
from dotenv import load_dotenv
from prompt_template import SYSTEM_PROMPT, RATING_PROMPT
from stt import get_transcript
import RPi.GPIO as GPIO


#GPIO setup
LED_PINS = {"green": 17, "yellow": 27, "red": 22}
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in LED_PINS.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)


load_dotenv()

def retrieve_response(prior_inputs, prior_responses, ordered_conversation):
    client = genai.Client()

    input_conversation = get_transcript()

    string_prior_inputs = ",\n".join(prior_inputs)
    string_prior_responses = ",\n".join(prior_responses)
    prompt = SYSTEM_PROMPT + "\nPrior inputs: " + string_prior_inputs + "\nPrior responses: " + string_prior_responses  + "\nInput sentence: " + input_conversation

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    response_string = response.text
    response_string = response.text[len('"AI Suggestion:'):] if "AI Suggestion: " in response.text else response.text
    response_string = response_string.replace('"', "")

    prior_inputs.append(input_conversation)
    prior_responses.append(response_string)
    ordered_conversation.append(input_conversation)
    ordered_conversation.append(response_string)
    print(response_string)
    return response_string

def get_convo_rating(conversation):
    client = genai.Client()
    string_conversation = ",\n".join(conversation)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=RATING_PROMPT + "\nConversation to this point: " + string_conversation,
    )
    rating_text = response.text
    print(rating_text)
    update_leds(rating_text)
    return rating_text
    
    
def update_leds(rating_text: str):
    # Turn off all LEDs first
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)

    rating = rating_text.strip().lower()
    if rating == "good":
        GPIO.output(LED_PINS["green"], GPIO.HIGH)
    elif rating == "ok":
        GPIO.output(LED_PINS["yellow"], GPIO.HIGH)
    elif rating == "bad":
        GPIO.output(LED_PINS["red"], GPIO.HIGH)

if __name__ == "__main__":
    previous_input_conversations = []
    previous_responses = []
    ordered_conversation = []

    try:
        while True:
            retrieve_response(previous_input_conversations, previous_responses, ordered_conversation)
            get_convo_rating(ordered_conversation)
    finally:
        GPIO.cleanup()