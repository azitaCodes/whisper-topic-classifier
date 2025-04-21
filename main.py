import os
import whisper
from pydub import AudioSegment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ✅ Step 1: Load Whisper model
print("🔧 Loading Whisper model...")
model = whisper.load_model("base")

# ✅ Step 2: Define your mp3 file
mp3_path = r"C:\Users\azira\audio_2025-04-20_19-07-15.mp3"
clean_wav_path = mp3_path.replace(".mp3", "_clean.wav")

# ✅ Step 3: Convert MP3 to clean WAV
print(f"\n🎧 Converting to WAV: {clean_wav_path}")
sound = AudioSegment.from_mp3(mp3_path)
sound.export(clean_wav_path, format="wav")
print("✅ Exported clean WAV")

# ✅ Step 4: Transcribe using Whisper
print("🧠 Transcribing with Whisper...")
result = model.transcribe(clean_wav_path)
text = result["text"].strip()

print("\n📝 Transcribed Text:\n", text)

# ✅ Step 5: Train a simple topic classifier
texts = [
    "We're making poached eggs and adding avocado",          # cooking
    "This dish uses rice, garlic, and lemon",                # cooking
    "Symptoms include fatigue, fever, and pain",             # health
    "Take two pills daily to treat infection",               # health
    "This tutorial teaches you how to code in Python",       # education
    "Artificial intelligence is transforming the tech world" # tech
]
labels = ["cooking", "cooking", "health", "health", "education", "tech"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
classifier = MultinomialNB()
classifier.fit(X, labels)

# ✅ Step 6: Predict the topic of your audio
input_vec = vectorizer.transform([text])
predicted_label = classifier.predict(input_vec)[0]

print(f"\n📌 Predicted Topic: {predicted_label}")