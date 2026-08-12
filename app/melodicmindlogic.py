from flask import Flask, render_template
import os
import csv

# Absolute Paths
TEMPLATE_FOLDER = r"C:\MelodicMind.AI\WEBSITE\templates"
STATIC_FOLDER = r"C:\MelodicMind.AI\WEBSITE\static"
SONG_FOLDER = r"C:\MelodicMind.AI\WEBSITE\songs"
SONG_CSV_PATH = r"C:\MelodicMind.AI\WEBSITE\song info.csv"
KARAOKE_CSV = r"C:\MelodicMind.AI\WEBSITE\karaoke.csv"
ARTIST_IMG_PATH = "/static/artists/"

# Initialize Flask
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/guitar')
def guitar():
    songs = []
    try:
        with open(SONG_CSV_PATH, newline='', encoding='utf-8-sig') as csvfile:
            raw = csv.reader(csvfile)
            headers = next(raw)
            headers = [h.strip().upper() for h in headers]
            reader = csv.DictReader(csvfile, fieldnames=headers)

            for row in reader:
                song = (row.get('SONG') or '').strip()
                artist = (row.get('ARTIST') or '').strip()
                year = (row.get('YEAR') or '').strip()

                if not song or not artist or not year:
                    continue

                filename = song + ".txt"
                file_path = os.path.join(SONG_FOLDER, filename)
                artist_img = artist.lower().replace(" ", "_") + ".jpg"

                if not os.path.exists(file_path):
                    continue

                songs.append({
                    'title': song.title(),
                    'artist': artist.title(),
                    'year': year,
                    'filename': filename,
                    'image': f"{ARTIST_IMG_PATH}{artist_img}"
                })

    except Exception as e:
        print(f"⚠️ Error loading song CSV: {e}")

    return render_template('guitar.html', songs=songs)

@app.route('/song/<song_name>')
def song_display(song_name):
    full_path = os.path.join(SONG_FOLDER, song_name)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"❌ Song not found: {song_name}"
    return render_template('song_display.html', lines=lines, song_name=song_name)

@app.route('/karaoke')
def karaoke():
    karaoke_files = []
    try:
        with open(KARAOKE_CSV, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                song = (row.get('SONG') or '').strip()
                if song:
                    karaoke_files.append(song)
    except Exception as e:
        return f"❌ Failed to load karaoke list: {e}"

    return render_template('karaoke.html', songs=karaoke_files)

@app.route('/karaoke/<song_name>')
def karaoke_display(song_name):
    youtube_url = None
    try:
        with open(KARAOKE_CSV, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                csv_song = row.get('SONG', '').strip().lower()
                match_song = song_name.replace('.txt', '').lower()

                if csv_song == match_song:
                    youtube_url = row.get('YOUTUBE_URL', '').strip()
                    if "watch?v=" in youtube_url:
                        youtube_url = youtube_url.replace("watch?v=", "embed/")
                    break
    except Exception as e:
        return f"❌ Error loading karaoke video: {e}"

    if not youtube_url:
        return f"❌ No video found for: {song_name}"

    return render_template(
        'karaoke_video.html',
        song_name=song_name.replace('.txt', '').title(),
        youtube_url=youtube_url
    )

@app.route('/exit')
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
        return "🛑 Server shutting down..."
    return "❌ Could not shut down."

# -------------------------
# 🎸 Chord Detection Section
# -------------------------
import sounddevice as sd
import librosa
import numpy as np
import joblib
import tensorflow as tf

MODEL_PATH = r"C:\MelodicMind.AI\MEL & MODEL\mel_cnn_model.h5"
ENCODER_PATH = r"C:\MelodicMind.AI\MEL & MODEL\label_encoder.pkl"

scroll_trigger_count = 0  # counter for valid chord predictions

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and encoder loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model or encoder: {e}")
    model = None
    encoder = None

def detect_chord_from_audio(duration=1.0, sr=22050):
    if model is None or encoder is None:
        print("❌ Model not loaded.")
        return False

    try:
        print("🎧 Listening for chord...")
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()

        y = np.squeeze(recording)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_db = mel_db[:, :128]
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')

        X = mel_db[np.newaxis, ..., np.newaxis]
        prediction = model.predict(X, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = encoder.inverse_transform([predicted_index])[0]
        confidence = np.max(prediction)

        print(f"🎯 Detected chord: {predicted_label} | Confidence: {confidence:.2f}")
        return confidence > 0.5
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return False

@app.route('/listen')
def listen():
    global scroll_trigger_count
    chord_detected = detect_chord_from_audio()

    if chord_detected:
        scroll_trigger_count += 1
        print(f"📈 Scroll counter: {scroll_trigger_count}/2")
        if scroll_trigger_count >= 2:
            scroll_trigger_count = 0
            return { "scroll": True }

    return { "scroll": False }

# -------------------------
# END
# -------------------------

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


