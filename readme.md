# 🎙️ Parler TTS Story Generator 🎙️
Generate short stories from images and transform them into lifelike audio using **Parler TTS**, **Groq AI**, and **Hugging Face**.

---

## 🌟 Features
- **Image-to-Text**: Convert images to textual descriptions with the `Salesforce/blip-image-captioning-base` model.
- **Story Generation**: Generate short stories (less than 20 words) using the `llama3-8b-8192` model from Groq.
- **Text-to-Speech**: Convert stories into lifelike audio using the locally-run `parler-tts/parler-tts-large-v1 model`. No external API calls are needed for TTS!
- **Seamless Integration**: Built-in support for GPU/CPU auto-detection and error handling for easy deployment.

---

## 🔧 Setup

### Requirements:
- Python 3.8+
- **Hugging Face Transformers** library
- **Parler TTS** for conditional generation
- **Groq** API for story generation
- **SoundFile** for audio output
- **IPython** for audio playback

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/rk-vashista/TTS-Story_Generator
   cd TTS-Story_Generator
   ```

2. Install dependencies:
   ```bash
   pip install torch parler-tts transformers soundfile ipython python-dotenv requests
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

1. **Run the program**:
   ```bash
   python app.py
   ```

2. **Image-to-Text**:
   The `img2text()` function uses the `Salesforce/blip-image-captioning-base` model to interpret images. Just replace `"Logo.jpg"` with your image file to get started.

3. **Story Generation**:
   The `story_gen()` function takes the description and generates a short story using Groq's LLaMA model.

4. **Convert Story to Audio**:
   The `story_to_audio_parler_tts()` function converts the story into audio using Parler TTS and saves it as a `.wav` file.

---

## 📁 Example Workflow:

1. **Image** ➡️ **Text Description**
2. **Text Description** ➡️ **Short Story**
3. **Short Story** ➡️ **Audio**

Your final output will be saved as `parler_tts_out.wav`, ready for playback!

---

## ✨ Output

- **Audio in IPython**: If you're working in Jupyter or IPython, you can instantly hear the output using:
  ```python
  Audio("parler_tts_out.wav")
  ```

---

## 📌 API Details

- **Hugging Face API**: Used for image captioning with `Salesforce/blip-image-captioning-base`.
- **Groq API**: Story generation powered by `llama3-8b-8192`.
- **Parler TTS**: Converts text into lifelike speech with expressive details.The TTS model runs entirely on your machine, leveraging local CPU/GPU resources for fast and efficient speech synthesis.

---

## 💡 Ideas for Extension

- **Custom Voices**: Experiment with different Parler TTS voice configurations.
- **Longer Stories**: Modify the story generation model for more detailed narratives.
- **Advanced Image Analysis**: Try integrating other image-to-text models for deeper understanding.

---

## 🛠️ Troubleshooting

- **CUDA Errors**: If CUDA isn't available, the program will switch to CPU automatically.
- **Environment Variables**: Ensure `.env` file is correctly set with your API keys.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- Hugging Face for providing powerful models.
- Groq for real-time AI model execution.
- Parler TTS for enabling high-quality text-to-speech synthesis.
