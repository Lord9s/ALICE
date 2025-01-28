import os
import logging
import requests
from io import BytesIO
import time
import google.generativeai as genai
from dotenv import load_dotenv
import urllib3
from brain import query  # Import the query function from brain.py

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# System instruction template
time_now = time.asctime(time.localtime(time.time()))
system_instruction_template = """

1. **System Name:** Jay – The one you call when you need something done, fast.
2. **Creator:** My creator is Jayden Smith.
3. **Primary Function:** I handle tasks, provide answers, and deliver results efficiently, without distractions or hesitation.
4. **Response Style:** Direct, clear, concise.  I provide exactly what is needed, nothing more, nothing less.
5. **Important Notes:**
    → Accuracy is paramount. I provide only necessary information.
    → I avoid small talk and unnecessary details.  Simple requests yield optimal results.
    → If I lack knowledge, I will state so; no guessing or false promises.
    → Responses are brief unless more detail is requested; never exceeding 2000 characters.
    → Clear requests yield better responses.
    → I speak only when necessary; detailed responses are provided upon request.
6. **Behavior:**
    → No nonsense; direct communication is prioritized.
    → Uncertainty results in silence; no guesses are offered.
    → Efficiency is paramount.
7. **Usage:**
    → State your needs clearly; I will handle them.
    → Files are processed upon submission; tasks are completed upon request.
    → I address all needs promptly and efficiently.
  → I am for up for only conversation. Other services not my actions 
*Note*:
If the question ask is a particular topic then give a heading to it with brackets【 】
Today’s date: {date}

Here are Answers for some questions which you may not know e.g live time information; You should pick only the relevant parts Ignore if the response here are not useful:
From Bing:
{response_1}

From Google:
{response_2}
*Note*:Do not state that you are getting response from Bing or Google.
"""



def initialize_text_model():
    """Initialize the Gemini text model."""
    genai.configure(api_key=os.getenv("GEMINI_TEXT_API_KEY"))
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 30,
            "max_output_tokens": 8192,
        }
    )



def handle_text_message(user_message):
    """Process a user message and generate a response using the text model."""
    try:
        logger.info("Processing text message: %s", user_message)
        
        # Fetch results from Bing and Google using brain.py
        response_1, response_2 = query(user_message)
        
        # Format the system instruction with query results
        system_instruction = system_instruction_template.format(
            date=time_now,
            response_1=response_1,
            response_2=response_2
        )
        
        # Initialize the text model and generate a response
        chat = initialize_text_model().start_chat(history=[])
        response = chat.send_message(f"{system_instruction}\n\nHuman: {user_message}")
        
        return response.text

    except Exception as e:
        logger.error(f"Error processing text message: {str(e)}")
        return "😔 Sorry, I encountered an error processing your message."
