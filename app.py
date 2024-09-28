import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, pipeline
import soundfile as sf
import requests
from dotenv import load_dotenv, find_dotenv
import os
from IPython.display import Audio
from groq import Groq  
# Load environment variables
load_dotenv(find_dotenv())

# Initialize the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize the Parler TTS model and tokenizer
try:
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
except RuntimeError as e:
    print("Failed to load model on GPU, switching to CPU:", e)
    device = "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)

tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

# Initialize the Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API"))  

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/myshell-ai/MeloTTS-English"
headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE')}"}

# Image interpretation
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0 if device == "cuda:0" else -1)
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


def story_to_audio_parler_tts(story_text, output_filename="parler_tts_out.wav"):
    description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch."

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(story_text, return_tensors="pt").input_ids.to(device)

    try:
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_filename, audio_arr, model.config.sampling_rate)
        print(f"Parler TTS Audio saved as {output_filename}")
    except RuntimeError as e:
        print("Error during audio generation with Parler TTS:", e)

# Main execution
def main():
    scenario = img2text("Logo.jpg")  # Replace with your image path
    story = story_gen(scenario)        # Story generation

    # Convert the generated story into audio using Parler TTS
    story_to_audio_parler_tts(story, output_filename="parler_tts_out.wav")

if __name__ == "__main__":
    main()
