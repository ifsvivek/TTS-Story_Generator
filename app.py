import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, pipeline
import soundfile as sf
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Load environment variables
load_dotenv(find_dotenv())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Initialize the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the Parler TTS model and tokenizer
@st.cache_resource
def load_model():
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-large-v1"
        ).to(device)
    except RuntimeError as e:
        st.write("Failed to load model on GPU, switching to CPU:", e)
        device = "cpu"
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-large-v1"
        ).to(device)
    return model

model = load_model()
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

# Initialize the Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API"))

# Image interpretation
def img2text(url):
    image_to_text = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=0 if device == "cuda:0" else -1,
    )
    result = image_to_text(url)
    text = result[0]["generated_text"] if result else ""
    return text

# Story generation using Groq API
def story_gen(scenario, model="llama3-8b-8192"):
    template = """
    You are a masterful storyteller:
    Your task is to craft short, impactful stories based on a given narrative. The story should be no longer than 20 words and must creatively reflect the essence of the provided scenario.
    Context: {scenario}
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
            },
        ],
        model=model,
    )
    story = chat_completion.choices[0].message.content
    return story

# Convert story to audio using Parler TTS
def story_to_audio_parler_tts(story_text, output_filename="parler_tts_out.wav"):
    description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch."

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(story_text, return_tensors="pt").input_ids.to(device)

    try:
        generation = model.generate(
            input_ids=input_ids, prompt_input_ids=prompt_input_ids
        )
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_filename, audio_arr, model.config.sampling_rate)
        return output_filename
    except RuntimeError as e:
        st.write("Error during audio generation with Parler TTS:", e)
        return None

# Streamlit UI
def main():
    st.title("Story to Audio Generator")

    # Upload an image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Image interpretation
        st.write("Generating text from the image...")
        image_url = uploaded_file
        scenario = img2text(image_url)  # Interpret image and generate text
        st.write("Image to Text Result:", scenario)

        # Story generation
        st.write("Generating story from the image description...")
        story = story_gen(scenario)
        st.write("Generated Story:", story)  # Display the generated story

        # Text-to-Speech conversion
        st.write("Converting the story to speech...")
        output_filename = story_to_audio_parler_tts(story)

        if output_filename:
            # Load and play the audio
            audio_file = open(output_filename, "rb").read()
            st.audio(audio_file, format="audio/wav")  # Media player for audio

            # Provide a download button for the audio
            st.download_button(label="Download the audio file", data=audio_file, file_name="parler_tts_out.wav")


if __name__ == "__main__":
    main()
