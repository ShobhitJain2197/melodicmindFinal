# 🎵 MelodicMind.AI

**MelodicMind.AI** is an AI-powered music learning platform designed to provide **real-time guitar chord recognition and interactive song practice** using deep learning and audio signal processing.

The project was developed as an end-to-end machine learning application, covering the complete pipeline from **dataset creation and augmentation** to **CNN model training, real-time audio inference, Flask-based deployment, and interactive web integration**.

> **Version 1 focuses primarily on guitar chord recognition and guided song practice.
> MelodicMind.AI Version 2 is currently under development as a significantly more advanced, production-oriented piano learning platform.**

---

## 🚀 Project Overview

The primary objective of MelodicMind.AI V1 was to build a system capable of listening to guitar audio through a microphone and automatically identifying the chord being played.

The detected chords are integrated with an interactive music-learning interface in which users can practice songs while the application listens to their playing and automatically advances through the lyrics/chord sheet.

### Core capabilities

* 🎸 Real-time guitar chord recognition
* 🎤 Live microphone-based audio inference
* 🧠 Deep Learning based chord classification
* 🎼 Mel-spectrogram feature extraction
* 📜 Automatic song-sheet scrolling
* 🎵 Guitar-oriented learning interface
* 🎬 Karaoke module with embedded YouTube playback
* 🌐 Flask-based web application
* 💻 Portable local deployment

---

# 🧠 Machine Learning Architecture

The chord recognition system uses a **Convolutional Neural Network (CNN)** implemented using **TensorFlow / Keras**.

The input audio is transformed into **Mel-spectrogram representations**, allowing the CNN to treat time-frequency information similarly to a two-dimensional image.

### Model Architecture

```text
Input Mel-Spectrogram
        ↓
Conv2D — 32 Filters
        ↓
Batch Normalization
        ↓
MaxPooling2D
        ↓
Dropout
        ↓
Conv2D — 64 Filters
        ↓
Batch Normalization
        ↓
MaxPooling2D
        ↓
Dropout
        ↓
Flatten
        ↓
Dense — 128 Neurons
        ↓
Dropout
        ↓
Softmax Output Layer
        ↓
Chord Classification
```

### Architecture Summary

| Component                    | Configuration             |
| ---------------------------- | ------------------------- |
| Convolutional Layers         | 2                         |
| Conv Filters                 | 32 → 64                   |
| Kernel Size                  | 3 × 3                     |
| Batch Normalization Layers   | 2                         |
| MaxPooling Layers            | 2                         |
| Fully Connected Hidden Layer | 128 neurons               |
| Output Layer                 | Softmax                   |
| Optimizer                    | Adam                      |
| Loss Function                | Categorical Cross-Entropy |
| Input Representation         | Mel Spectrogram           |
| Framework                    | TensorFlow / Keras        |

---

# 📊 Model Performance

The final trained CNN achieved approximately:

## **74.1% Validation Accuracy**

This result was achieved despite working with a relatively small custom audio dataset.

The primary challenge was therefore not only model design, but also improving generalization from a limited number of original guitar recordings.

To address this, an extensive **audio data augmentation pipeline** was implemented.

---

# 🎛️ Dataset Creation

A custom guitar chord dataset was created specifically for the project.

Each audio recording represented an individual guitar chord rather than an entire song.

The recordings were converted into Mel-spectrograms before being passed to the CNN.

The dataset pipeline included:

```text
Raw Guitar Audio
       ↓
Audio Preprocessing
       ↓
Data Augmentation
       ↓
Mel Spectrogram Generation
       ↓
Label Encoding
       ↓
CNN Training
```

---

# 🔄 Audio Data Augmentation

Because the original dataset was relatively small, several augmentation techniques were applied to synthetically increase dataset diversity.

For each original audio sample, multiple modified versions were generated.

### Augmentation techniques

* Original recording
* Positive pitch shift
* Negative pitch shift
* Faster time-stretching
* Slower time-stretching
* Gaussian / background noise injection
* Volume / amplitude modification

Approximately **7 versions of each original sample** were generated.

This increased the effective training dataset while helping the model become more robust to real-world differences in guitar playing and recording conditions.

### Why augmentation was important

Without augmentation, the CNN could easily memorize:

* Specific recording conditions
* Exact pitch characteristics
* Microphone behaviour
* Guitar tone
* Playing intensity

Augmentation introduced controlled variations and improved the model's ability to generalize toward unseen audio.

---

# 🎼 Mel-Spectrogram Feature Extraction

Raw audio waveforms were transformed into **Mel-spectrograms** before model training.

A Mel-spectrogram represents:

* **X-axis:** Time
* **Y-axis:** Frequency
* **Intensity:** Signal energy at a particular frequency and time

The model used spectrogram inputs approximately structured as:

```text
128 × 128 × 1
```

where the final dimension represents a single-channel spectrogram.

This allowed the CNN to learn patterns corresponding to:

* Harmonic structure
* Frequency distribution
* Chord-specific spectral characteristics
* Guitar resonance
* Energy distribution across frequency bands

---

# 🎯 Real-Time Chord Recognition

After training, the model was integrated into the web application for live inference.

The application:

1. Records approximately one second of microphone audio.
2. Converts the recording into a Mel-spectrogram.
3. Resizes / pads the spectrogram into the expected model input shape.
4. Sends it to the trained CNN.
5. Obtains chord probabilities through the Softmax output layer.
6. Applies a confidence threshold.
7. Returns the predicted chord to the application.

A confidence threshold of approximately:

```text
0.50
```

was used to reject uncertain predictions.

---

# 📜 Automatic Song Scrolling

One of the primary interactive features of MelodicMind.AI is **audio-driven automatic lyric scrolling**.

Instead of requiring the user to manually scroll while playing guitar, the application continuously listens for chord predictions.

The simplified logic is:

```text
User Plays Guitar
       ↓
Microphone Captures Audio
       ↓
CNN Predicts Chord
       ↓
Valid Prediction Detected
       ↓
Prediction Counter Updated
       ↓
Two Consecutive Valid Chords
       ↓
Automatically Scroll Song Sheet
```

This provides a more natural hands-free practice experience.

---

# 🌐 Web Application

The application was built using:

* **Python**
* **Flask**
* **HTML**
* **CSS**
* **JavaScript**
* **CSV-based song metadata**
* **TensorFlow / Keras**

The Flask backend handles:

* Model loading
* Audio recording
* Spectrogram generation
* Inference
* Song routing
* Chord-triggered scrolling
* Karaoke content loading

---

# 🎸 Guitar Module

The guitar section allows users to select songs and practice using chord sheets.

Song data includes:

* Song title
* Artist information
* Chords
* Lyrics
* Tuning
* Capo information
* Song sections such as Verse / Chorus / Intro

The song pages dynamically load the required information and provide automatic scrolling based on live chord recognition.

---

# 🎬 Karaoke Module

Version 1 also includes a lightweight karaoke system.

The karaoke section uses:

* Song metadata stored in CSV files
* YouTube video URLs
* Embedded video playback
* Dynamically generated karaoke pages

The karaoke module was designed as an additional music-learning feature alongside the primary guitar chord-recognition system.

---

# 🗂️ Project Structure

A simplified representation of the project is:

```text
MelodicMind.AI/
│
├── MelodicMindfull.ipynb
│
├── melodicmindlogic.py
│
├── index.html
├── guitar.html
├── song_display.html
│
├── karaoke.html
├── karaoke_display.html
├── karaoke_video.html
│
├── song info.csv
├── karaoke.csv
│
├── style.css
│
└── Model / Dataset Assets
```

---

# 🛠️ Technology Stack

### Machine Learning

* Python
* TensorFlow
* Keras
* Scikit-learn
* NumPy
* Librosa

### Deep Learning

* Convolutional Neural Networks
* Batch Normalization
* Dropout Regularization
* Softmax Classification

### Audio Processing

* Mel Spectrograms
* Pitch Shifting
* Time Stretching
* Noise Injection
* Amplitude Augmentation

### Backend

* Flask
* Python

### Frontend

* HTML
* CSS
* JavaScript

### Data Management

* CSV
* NumPy Arrays
* Pickle-based Label Encoding

---

# 💾 Model Artifacts

The trained system uses several serialized assets including:

```text
mel_cnn_model.h5
label_encoder.pkl
mel_spectrograms.npy
mel_labels.npy
```

These allow the trained model and label mappings to be reused during inference without retraining.

---

# 📈 Training Strategy

The model was trained using:

* Adam optimizer
* Categorical cross-entropy loss
* Mini-batch training
* Validation monitoring
* Early stopping
* Dropout regularization
* Batch normalization

Early stopping was used to monitor validation loss and prevent unnecessary training once generalization performance stopped improving.

The best-performing model weights were restored automatically.

---

# 🔬 Key Machine Learning Concepts Demonstrated

This project demonstrates practical implementation of:

* Supervised Learning
* Multi-Class Classification
* Convolutional Neural Networks
* Audio Signal Processing
* Mel-Frequency Representations
* Feature Extraction
* Data Augmentation
* Regularization
* Batch Normalization
* Early Stopping
* Real-Time Model Inference
* Confidence Thresholding
* End-to-End ML Deployment

---

# 💡 Challenges Addressed

### Limited Dataset Size

The project originally contained a relatively small number of guitar recordings.

This was addressed using extensive audio augmentation to create multiple training examples from each original sample.

### Overfitting

Regularization techniques were incorporated, including:

* Dropout
* Batch Normalization
* Data Augmentation
* Early Stopping

### Real-Time Integration

The trained CNN was successfully integrated into a web application capable of receiving live microphone recordings and generating predictions during user practice.

---

# 🏗️ End-to-End Architecture

```text
                    MelodicMind.AI
                           │
           ┌───────────────┴───────────────┐
           │                               │
      Guitar Module                  Karaoke Module
           │                               │
      Song Selection                   Song Selection
           │                               │
       Chord Sheet                    YouTube Playback
           │
       Microphone
           │
      Audio Capture
           │
     Mel Spectrogram
           │
       CNN Model
           │
    Chord Prediction
           │
 Confidence Filtering
           │
 Automatic Scrolling
```

---

# 🚀 MelodicMind.AI Version 2 — Coming Soon

MelodicMind.AI is currently being **rebuilt from the ground up** as a significantly more advanced and product-oriented platform.

Version 2 will shift the primary focus from guitar to **piano learning** and is being designed as a complete commercial-grade learning experience rather than only a machine-learning demonstration.

### Planned Version 2 capabilities

* 🎹 MIDI keyboard integration
* 🎼 Real-time piano note detection
* 🎵 Falling-note / piano-roll visualization
* 🎯 Real-time note accuracy measurement
* ⏱️ Timing and tempo analysis
* ❌ Wrong-note and missed-note detection
* 📊 Detailed performance scoring
* 📈 User progress analytics
* 🧠 AI-generated practice feedback
* 🎓 Structured piano learning curriculum
* 🎶 Built-in MIDI song library
* 🐢 Slow / Normal / Fast practice modes
* ⚡ Adaptive tempo training
* 🏆 Performance rankings and progression
* 🌐 Production-grade web application
* 📱 Dedicated mobile application

---

## Future AI Research Direction

Future versions are planned to explore more advanced machine-learning architectures including:

* Audio Spectrogram Transformers
* Transformer-based audio classification
* CNN-BiLSTM architectures
* Attention mechanisms
* Music Information Retrieval
* Audio-to-MIDI transcription
* Personalized recommendation systems
* Adaptive learning models
* Large Language Model based AI coaching

The long-term goal is to evolve MelodicMind.AI into an intelligent piano-learning platform capable of understanding how a musician plays, identifying weaknesses, and creating personalized practice recommendations.

---

# 📌 Version Status

```text
MelodicMind.AI V1
Status: Completed ✅

MelodicMind.AI V2
Status: In Development 🚧
```

---

# 🎯 Project Objective

MelodicMind.AI demonstrates how **Deep Learning, Digital Signal Processing and Full-Stack Development** can be combined to create an interactive real-world music-learning application.

The project evolved from a custom guitar chord classification experiment into a broader vision for an **AI-powered personalized music education platform**.

---

## ⭐ MelodicMind.AI

**Learn. Play. Improve with AI.**
