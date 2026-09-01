# Air Canvas — Virtual Pen

Draw in the air by moving a colored object in front of your webcam. Built
with OpenCV (HSV color tracking + contour detection) and a tkinter control
panel — no Flask, no MediaPipe, no deep learning framework required for
core drawing.

An optional add-on recognizes hand-drawn letters (A–Z) using a lightweight
scikit-learn classifier trained on EMNIST.

## Features

- Real-time marker tracking via HSV color thresholding + contour detection
- tkinter control panel: color picker, brush size, eraser, clear canvas
- Live HSV calibration sliders (tune tracking to your marker/lighting)
- Optional: recognize a single drawn letter (A–Z) with one click

## Demo

| Draw | Recognize |
|---|---|
| Move a colored marker to draw/erase on screen | Click "Recognize Letter" to classify what you drew |

## Requirements

- Python 3.8+
- tkinter (bundled with Python; on Linux install separately if missing — see below)

```bash
pip install -r requirements.txt
```

`opencv-python` and `numpy` are required for the core app. `pandas`,
`scikit-learn`, and `joblib` are only needed if you want letter
recognition.

### tkinter on Linux
If `python3 -c "import tkinter"` errors out:
```bash
sudo apt install python3-tk        # Ubuntu/Debian
```

## Usage

```bash
python air_canvas.py
```

- Hold up a solid-colored object (blue by default) in front of your webcam.
- Use the **Marker Mask** window to check tracking — your object should
  appear as a clean white blob on a black background. Adjust the HSV
  sliders in the control panel if it doesn't.
- Draw by moving the object; pick colors/eraser/brush size from the panel.
- Press `c` (clear) or `q` (quit) in the camera window, or use the panel buttons.

## Letter Recognition (optional)

The "Recognize Letter" button needs a trained model first. This is a
one-time setup:

1. Install the extra dependencies (included in `requirements.txt`):
   `pandas`, `scikit-learn`, `joblib`
2. Download the EMNIST letters dataset from Kaggle:
   https://www.kaggle.com/datasets/crawford/emnist
   Extract `emnist-letters-train.csv` and `emnist-letters-test.csv` into
   the project folder.
3. Run the training script once:
   ```bash
   python train_letter_classifier.py
   ```
   This saves `letter_model.joblib` in the project folder. `air_canvas.py`
   loads it automatically on startup.

Without this file, the app still runs fine — the "Recognize Letter" button
will just say the model isn't available.

## Project Structure

```
air-canvas/
├── air_canvas.py               # Main application (run this)
├── train_letter_classifier.py  # One-time letter-recognition training script
├── requirements.txt
├── .gitignore
└── README.md
```

## How the tracking works

1. Convert each webcam frame to HSV.
2. Threshold against a color range to get a binary mask of the marker.
3. Find the largest contour in the mask — that's the marker tip.
4. Track the tip's center across frames, connecting points into strokes
   drawn onto a persistent canvas overlaid on the live feed.

No hand-landmark model, no neural network for tracking — just classic
OpenCV contour tracking, so it runs fast on modest hardware.

## Notes

- The trained model (`letter_model.joblib`) and the EMNIST CSVs are not
  included in this repo (see `.gitignore`) since they're large / regenerable.
  Follow the steps above to create them locally.
- Recognition accuracy depends on how cleanly a letter is drawn — air-drawn
  strokes are blobbier than the scanned handwriting EMNIST was trained on,
  so it works best on clear, single, block letters.

## License

MIT (or your preferred license — add a `LICENSE` file)
