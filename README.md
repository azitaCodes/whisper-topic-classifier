# 🎧 Whisper Voice Topic Classifier

This project takes an English voice audio file and:

1. Converts the audio to text using OpenAI's Whisper
2. Predicts the topic (like cooking, health, education, or tech) using machine learning

---

## ✅ How It Works

- MP3 audio is converted to clean `.wav` format
- Whisper transcribes the audio to text
- A Naive Bayes classifier guesses what topic the text is about

---

## 🧪 Sample Topics

- Cooking
- Health
- Education
- Tech

---

## 💻 Requirements

Install them with:

```bash
pip install git+https://github.com/openai/whisper.git
pip install pydub scikit-learn ffmpeg-python
