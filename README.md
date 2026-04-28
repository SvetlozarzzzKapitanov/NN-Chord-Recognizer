# Real-Time Guitar Chord Recognition using Neural Networks

This project implements a real-time guitar chord recognition system using deep learning techniques. It processes audio input, extracts relevant features, and predicts chords using a trained neural network model.

---

## Features

- Real-time chord recognition from microphone input  
- Offline prediction from `.wav` audio files  
- Neural network trained on labeled chord dataset  
- Supports major and minor chords  
- Simple GUI for live interaction  
- Improved prediction stability using aggregation strategy  

---

## How It Works

The system follows a standard machine learning pipeline:

1. **Audio Processing**
   - Audio is captured from a microphone or loaded from a `.wav` file
   - Preprocessing is performed using `librosa`

2. **Feature Extraction**
   - Audio is converted into numerical representations (e.g. chroma features)

3. **Model Prediction**
   - A trained neural network outputs chord probabilities

4. **Aggregation (Real-Time Stability)**
   - Multiple predictions are combined over time
   - A max-based aggregation strategy is used instead of mean to improve stability

5. **Output**
   - The most probable chord is displayed in real time

---

## Project Structure

## Project Structure

```
NN-Chord-Recognizer/
├── src/                     # Core application logic
│   ├── features.py          # Feature extraction
│   ├── train.py             # Model training
│   ├── predict.py           # Offline prediction
│   ├── realtime_predict.py  # Real-time inference
│   └── gui_app.py           # GUI application
│
├── data/                    # Training dataset (organized by chord)
│   ├── A/
│   ├── Am/
│   ├── B/
│   ├── Bm/
│   ├── C/
│   ├── Cm/
│   ├── D/
│   ├── Dm/
│   ├── E/
│   ├── Em/
│   ├── F/
│   ├── Fm/
│   ├── G/
│   └── Gm/
│
├── demo_samples/            # Example .wav files for testing
│   ├── A.wav
│   ├── Am.wav
│   ├── ...
│
├── models/                  # Trained models
│   └── chord_mlp.pt
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Model Details

- Architecture: Neural Network (feedforward)
- Training:
  - Supervised learning on labeled chord data
- Loss Function:
  - Cross-entropy (classification)
- Optimization:
  - Gradient descent-based methods

The model learns patterns in extracted audio features and maps them to chord classes through nonlinear transformations.

---

## Limitations

- Real-time performance depends on microphone quality  
- Some chords may be misclassified (e.g. major vs minor)  
- Background noise can reduce accuracy  

---

## Future Improvements

- Expand dataset (more samples per chord)
- Add support for additional chord types (e.g. 7th, suspended)
- Improve robustness to noise
- Improve the design of confidence visualization and prediction history
- Explore mobile or web deployment

---

## Technologies Used

- Python
- PyTorch
- Librosa
- NumPy
- Tkinter

---