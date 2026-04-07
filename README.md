# Real-Time Guitar Chord Recognition

## Overview

This project implements a real-time guitar chord recognition system using a neural network trained on audio features extracted from chord recordings.

The system supports:

* Offline training on labeled chord datasets
* Real-time chord prediction using microphone input
* GUI-based visualization of predictions and confidence
* Demo mode using pre-recorded audio samples

---

## Features

* Chord classification (major and minor)
* Real-time prediction with smoothing and confidence filtering
* Confusion matrix visualization for evaluation
* Interactive GUI
* Demo playback mode (pre-recorded audio → model input)

---

## How It Works

1. Audio is converted into chroma-based features (mean + std)
2. Features are normalized using training statistics
3. A small MLP neural network predicts chord probabilities
4. Predictions are smoothed over time (max-based aggregation)
5. GUI displays:

   * Detected chord
   * Confidence
   * Top-3 predictions

---

## Running the App

### 1. Train the model

```bash
python train.py
```

### 2. Launch GUI

```bash
python gui_app.py
```

---

## Demo Mode (Recommended for Presentations)

The GUI includes buttons that:

* Play pre-recorded chord audio
* Feed the same audio into the model
* Display predictions in real time

This provides a stable and reproducible demonstration environment.

### Setup

Create a folder:

```
demo_samples/
```

Add `.wav` files (e.g.):

```
C.wav
G.wav
Am.wav
F.wav
D.wav
Em.wav
```

---

## Limitations

* Performance depends heavily on audio quality
* Real-time predictions may degrade with:

  * background noise
  * low-quality microphones
  * compressed audio (e.g., phone playback)
* Chroma features may struggle with major/minor distinctions

---

## TODO

* [ ] Record demo samples for presentation

---

## Notes

The system performs best when the input audio closely matches the training data conditions. Demo mode is recommended for consistent results during presentations.
