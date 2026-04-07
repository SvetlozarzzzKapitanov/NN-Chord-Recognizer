import tkinter as tk
from tkinter import ttk, messagebox

import threading
import time
import numpy as np
import librosa
import sounddevice as sd

from realtime_predict import RealTimeChordRecognizer


class ChordRecognizerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Real-Time Chord Recognizer")
        self.root.geometry("1280x768")
        self.root.resizable(True, True)

        try:
            self.recognizer = RealTimeChordRecognizer()
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            raise

        self._build_ui()
        self._schedule_update()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="Real-Time Chord Recognizer",
            font=("Segoe UI", 18, "bold")
        )
        title.pack(pady=(0, 20))

        self.status_var = tk.StringVar(value="Status: Idle")
        self.detected_var = tk.StringVar(value="Press Start\nListening")
        self.raw_var = tk.StringVar(value="Raw: -")
        self.confidence_var = tk.StringVar(value="Confidence: 0.000")
        self.energy_var = tk.StringVar(value="Signal energy: 0.0000")
        self.top_var = tk.StringVar(value="Top3: -")

        ttk.Label(main, textvariable=self.status_var, font=("Segoe UI", 11)).pack(pady=(0, 10))

        ttk.Label(main, text="Detected Chord", font=("Segoe UI", 12)).pack()

        ttk.Label(
            main,
            textvariable=self.detected_var,
            font=("Segoe UI", 28, "bold")
        ).pack(pady=(5, 15))

        ttk.Label(main, textvariable=self.raw_var, font=("Segoe UI", 10)).pack()
        ttk.Label(main, textvariable=self.confidence_var, font=("Segoe UI", 10)).pack()
        ttk.Label(main, textvariable=self.energy_var, font=("Segoe UI", 10)).pack()

        ttk.Label(
            main,
            textvariable=self.top_var,
            font=("Segoe UI", 10),
            wraplength=440,
            justify="center"
        ).pack(pady=(8, 18))

        # ================= Buttons =================

        buttons = ttk.Frame(main)
        buttons.pack(pady=(0, 10))

        self.start_button = ttk.Button(buttons, text="Start Listening", command=self.start_listening)
        self.start_button.grid(row=0, column=0, padx=8)

        self.stop_button = ttk.Button(buttons, text="Stop Listening", command=self.stop_listening)
        self.stop_button.grid(row=0, column=1, padx=8)

        # ================= Demo buttons =================

        demo_frame = ttk.LabelFrame(main, text="Demo (Pre-recorded Chords)")
        demo_frame.pack(pady=10)

        chords = ["C", "G", "Am", "F", "D", "Em"]

        for i, chord in enumerate(chords):
            ttk.Button(
                demo_frame,
                text=f"Play {chord}",
                command=lambda c=chord: self.play_and_predict(f"../demo_samples/{c}.wav")
            ).grid(row=0, column=i, padx=5, pady=5)

    # ================= Demo Logic =================

    def play_and_predict(self, path):
        def worker():
            try:
                y, sr = librosa.load(path, sr=self.recognizer.sr, mono=True)

                # Play audio so audience hears it
                sd.play(y, sr)

                chunk_size = self.recognizer.hop_samples

                for i in range(0, len(y), chunk_size):
                    chunk = y[i:i + chunk_size]

                    self.recognizer.feed_audio_chunk(chunk)
                    self.recognizer.process_buffer()

                    time.sleep(self.recognizer.hop_seconds)

            except Exception as e:
                messagebox.showerror("Playback Error", str(e))

        threading.Thread(target=worker, daemon=True).start()

    # ================= Controls =================

    def start_listening(self):
        try:
            self.recognizer.start()
        except Exception as e:
            messagebox.showerror("Start Error", str(e))

    def stop_listening(self):
        try:
            self.recognizer.stop()
        except Exception as e:
            messagebox.showerror("Stop Error", str(e))

    # ================= UI Update =================

    def _schedule_update(self):
        self._update_ui()
        self.root.after(100, self._schedule_update)

    def _update_ui(self):
        state = self.recognizer.get_state()

        self.status_var.set(f"Status: {state['status']}")
        self.detected_var.set(state["detected_label"])
        self.raw_var.set(f"Raw: {state['raw_label']}")
        self.confidence_var.set(f"Confidence: {state['confidence']:.3f}")
        self.energy_var.set(f"Signal energy: {state['energy']:.4f}")
        self.top_var.set(f"Top3: {state['top_text']}")

        if state["is_running"]:
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
        else:
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.detected_var.set("Press 'Start Listening' to detect chords")

    def on_close(self):
        try:
            self.recognizer.stop()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    app = ChordRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()