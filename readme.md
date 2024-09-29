
# 🎙️ Parler TTS Story Generator 🎙️
Generate short stories from images and transform them into lifelike audio using **Parler TTS**, **Groq AI**, and **Hugging Face** models, all within an interactive **Streamlit** app.

---

## 🌟 Features
- **Image-to-Text Interpretation**: Convert images to textual descriptions with the `Salesforce/blip-image-captioning-base` model.
- **Story Generation**: Generate short, creative stories (less than 20 words) based on the interpreted text using the `llama3-8b-8192` model from Groq.
- **Text-to-Speech**: Convert generated stories into lifelike audio using the locally-run `parler-tts/parler-tts-large-v1` model, leveraging your GPU/CPU for fast processing.
- **Streamlit Integration**: Interactive UI for uploading images, generating stories, and downloading audio.
- **Error Handling**: Automatically switches to CPU if GPU isn't available and handles model loading errors.

---

## 🔧 Setup

### Requirements:
- Python 3.8+
- **Hugging Face Transformers** library
- **Parler TTS** for conditional generation
- **Groq** API for story generation
- **SoundFile** for audio output
- **Streamlit** for the UI

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/rk-vashista/TTS-Story_Generator
   cd TTS-Story_Generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file:
     ```bash
     touch .env
     ```
   - Add your API keys to `.env`:
     ```bash
     HUGGING_FACE=<your_hugging_face_api_key>
     GROQ_API=<your_groq_api_key>
     ```

---

## 🚀 Usage

1. **Run the app**:
   ```bash
   streamlit run app.py
   ```

2. **Upload an image**:
   Upload a `.jpg`, `.png`, or `.jpeg` image to generate a scenario description.

3. **Generate a short story**:
   The app will automatically interpret the uploaded image and generate a short story using Groq's LLaMA model.

4. **Convert the story into audio**:
   The app will convert the story into a `.wav` audio file using Parler TTS, which you can listen to and download.

---

## 📁 Example Workflow:

1. **Upload Image** ➡️ **Text Description**
2. **Text Description** ➡️ **Short Story**
3. **Short Story** ➡️ **Lifelike Audio**

The final audio will be saved as `parler_tts_out.wav`, available for download directly through the app.

---

## 🔑 Environment Variables

Create a `.env` file in the project root to store your API keys:
```
HUGGING_FACE=<your_hugging_face_api_key>
GROQ_API=<your_groq_api_key>
```

---

## 📌 API Details

- **Hugging Face API**: Used for image captioning with the `Salesforce/blip-image-captioning-base` model.
- **Groq API**: Powers the story generation using `llama3-8b-8192`.
- **Parler TTS**: Converts text into expressive speech using the `parler-tts-large-v1` model, running on local resources.

---

## ✨ Possible Extensions

- **Custom Voices**: Experiment with different Parler TTS voice settings or modify the description to adjust the voice's style.
- **Longer Stories**: Adapt the story generation prompt to produce more detailed narratives.
- **Advanced Image Analysis**: Try other image-to-text models for deeper interpretation of the uploaded images.

---

## 🛠️ Troubleshooting

- **CUDA Errors**: If CUDA isn't available, the program will switch to CPU automatically.
- **Environment Variables**: Ensure the `.env` file contains correct API keys for Hugging Face and Groq.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- Hugging Face for providing powerful models.
- Groq for real-time AI model execution.
- Parler TTS for enabling high-quality text-to-speech synthesis.
- Streamlit for providing a seamless UI for deployment.