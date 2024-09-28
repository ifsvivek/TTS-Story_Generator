from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from groq import Groq
import os
import soundfile as sf  # To save audio data
import numpy as np
import requests
from IPython.display import Audio

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Groq API client
groq_client = Groq(api_key=os.getenv("GROQ_API"))

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE')}"}

# Image interpretation
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result = image_to_text(url)
    text = result[0]['generated_text'] if result else ""
    print("Image to text result:", text)
    return text

# Story generation using Groq API
def story_gen(scenario, model="llama3-8b-8192"):
    template = """
    You are a story teller:
    You can generate a short story based on a simple narrative, the story should be no more than 20 words.
    CONTEXT: {scenario}
    """
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": template.format(scenario=scenario),
            },
            {
                "role": "user",
                "content": scenario,
            }
        ],
        model=model,
    )
    story = chat_completion.choices[0].message.content
    print("Generated story:", story)
    return story

# Text-to-Speech conversion using Hugging Face API
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def story_to_audio(story_text, output_filename="generated_story.mp3"):
    audio_bytes = query({"inputs": story_text})
    
    # Convert the raw audio data to MP3 format without installing extra packages
    with open(output_filename, "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved as {output_filename}")

    # Play the audio
    return Audio(audio_bytes)

# Main execution
scenario = img2text("Logo.jpg")  # Image interpretation
story = story_gen(scenario)      # Story generation

# Convert the generated story into audio
audio_player = story_to_audio(story, output_filename="generated_story.mp3")  # Save and play story as audio

