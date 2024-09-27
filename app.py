from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from groq import Groq
import os
import time  # Import time for sleep functionality


# Load environment variables
load_dotenv(find_dotenv())

# Initialize Groq API client
groq_client = Groq(api_key=os.getenv("GROQ_API"))

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
    You are a story Teller:
    You can generate a short story based on a simple narrative, the story should be no more than 200 words.
    CONTEXT: {scenario}
    """
    # Create chat completion
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": template.format(scenario=scenario),  # Correctly format the template
            },
            {
                "role": "user",
                "content": scenario,  # User's scenario
            }
        ],
        model=model,
    )
    # Return the content of the completion
    story = chat_completion.choices[0].message.content
    print("Generated story:", story)
    return story

# Text-to-Speech from generated story
def story_to_audio(story_text, output_filename="generated_story.mp3"):
    tts = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")
    tts_result = tts(story_text)
    # Save the audio to an MP3 file
    with open(output_filename, "wb") as f:
        f.write(tts_result[0]["audio"])
    print(f"Audio saved as {output_filename}")

# Main execution
scenario = img2text("Logo.jpg")  # Image interpretation
story = story_gen(scenario)      # Story generation

# Convert the generated story into audio
story_to_audio(story, output_filename="generated_story.mp3")  # Save story to audio
