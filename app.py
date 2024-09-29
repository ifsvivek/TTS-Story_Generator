import torch
import base64
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, pipeline
import soundfile as sf
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from groq import Groq
from io import BytesIO

# Load environment variables
load_dotenv(find_dotenv())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the Parler TTS model and tokenizer


@st.cache_resource
def load_model():
    global device
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-large-v1"
        ).to(device)
    except RuntimeError as e:
        st.error(f"Failed to load model on GPU, switching to CPU: {e}")
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

    input_ids = tokenizer(
        description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(
        story_text, return_tensors="pt").input_ids.to(device)

    try:
        generation = model.generate(
            input_ids=input_ids, prompt_input_ids=prompt_input_ids
        )
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_filename, audio_arr, model.config.sampling_rate)
        return output_filename
    except RuntimeError as e:
        st.error(f"Error during audio generation with Parler TTS: {e}")
        return None

# Streamlit UI


def main():

    st.title("Story to Audio Generator")
    st.write(
        "Generate a short story based on an image and convert it to audio using AI!")

    # Upload an image file
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image",
                 use_column_width=True)

    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            # Convert image to text
            st.subheader("Step 1: Image to Text Interpretation")
            image_url = uploaded_file
            image = base64.b64encode(image_url.read()).decode("utf-8")
            scenario = img2text(image)  # Interpret image and generate text
            st.success("Image to Text Result:")
            st.write(f"**Scenario:** {scenario}")

            # Story generation
            st.subheader("Step 2: Story Generation")
            story = story_gen(scenario)
            st.success("Generated Story:")
            st.write(f"**Story:** {story}")

        with st.spinner("Converting story to speech..."):
            # Text-to-Speech conversion
            st.subheader("Step 3: Text to Speech Conversion")
            output_filename = story_to_audio_parler_tts(story)

            if output_filename:
                # Load and play the audio
                with open(output_filename, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    # Media player for audio
                    st.audio(audio_bytes, format="audio/wav")

                # Provide a download button for the audio
                st.download_button(
                    label="Download the audio file",
                    data=audio_bytes,
                    file_name="parler_tts_out.wav",
                    mime="audio/wav"
                )
            else:
                st.error(
                    "There was an issue generating the audio. Please try again.")


if __name__ == "__main__":
    main()
