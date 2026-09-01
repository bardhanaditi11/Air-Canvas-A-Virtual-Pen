"""
Air Canvas - Virtual Pen
-------------------------
Draw in the air by moving a colored object (default: a blue marker/cap)
in front of your webcam. A tkinter control panel lets you pick draw
colors, change brush size, clear the canvas, and calibrate the color
tracker live.

Only 2 external dependencies: opencv-python and numpy.
tkinter is part of the Python standard library (no pip install needed),
though on some Linux distros it ships as a separate OS package:
    sudo apt install python3-tk

No Flask, no mediapipe, no web server - everything runs in one process.

How the tracking works (classic, lightweight approach):
    1. Convert each webcam frame to HSV.
    2. Threshold it against a color range (defaults tuned for a blue
       object) to get a binary mask of "the marker".
    3. Find the largest contour in that mask -> that's the marker tip.
    4. Track the tip's center across frames and connect the dots with
       lines on a persistent canvas overlay.
No neural network, no landmark model - just OpenCV contour tracking,
so it runs fast even on modest hardware.

Controls (in the tkinter window):
    - Color swatches: Blue / Green / Red / Yellow / Eraser
    - Brush size slider
    - Clear Canvas button
    - HSV calibration sliders (to match your marker/lighting)
    - Recognize Letter button (optional - see below)
    - Quit button

Controls (in the OpenCV video window):
    - 'c' key: clear canvas
    - 'q' key or closing the window: quit

Optional: letter recognition
    "Recognize Letter" reads whatever single letter is currently drawn
    on the canvas and predicts it, using a small scikit-learn model
    (MLPClassifier trained on EMNIST). This needs a one-time training
    step first:
        pip install emnist scikit-learn joblib
        python train_letter_classifier.py
    That produces letter_model.joblib next to this script. If that
    file isn't present, the button just tells you it's not available -
    drawing/erasing still works fine without it.
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from collections import deque

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480
MAX_POINTS = 1024          # how long a single stroke's trail can get
MIN_CONTOUR_AREA = 400     # ignore tiny noise blobs in the mask

DRAW_COLORS = {
    "Blue":   (255, 0, 0),
    "Green":  (0, 255, 0),
    "Red":    (0, 0, 255),
    "Yellow": (0, 255, 255),
}

# Default HSV range tuned for a blue marker cap; adjust with sliders
DEFAULT_HSV_LOWER = [94, 80, 2]
DEFAULT_HSV_UPPER = [126, 255, 255]


class AirCanvas:
    def __init__(self):
        # --- video capture ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")

        # --- persistent drawing surface (same size as the frame) ---
        self.canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        # --- drawing state ---
        self.points = deque(maxlen=MAX_POINTS)   # None = pen lifted (stroke break)
        self.draw_color = DRAW_COLORS["Blue"]
        self.brush_size = 6
        self.eraser_on = False

        # --- optional letter-recognition model ---
        self.letter_model = None
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "letter_model.joblib")
        if _JOBLIB_AVAILABLE and os.path.exists(model_path):
            try:
                self.letter_model = joblib.load(model_path)
            except Exception as e:
                print(f"Could not load letter_model.joblib: {e}")

        # --- tkinter control panel ---
        self.root = tk.Tk()
        self.root.title("Air Canvas - Controls")
        self.root.geometry("300x420")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build_controls()

        cv2.namedWindow("Air Canvas - Camera")

    # ------------------------------------------------------------------
    # tkinter UI
    # ------------------------------------------------------------------
    def _build_controls(self):
        pad = {"padx": 10, "pady": 6}

        ttk.Label(self.root, text="Draw Color", font=("Segoe UI", 11, "bold")).pack(**pad)

        color_frame = tk.Frame(self.root)
        color_frame.pack(**pad)
        for name, bgr in DRAW_COLORS.items():
            hexcolor = "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])  # BGR -> RGB hex
            tk.Button(
                color_frame, text=name, bg=hexcolor,
                fg="white" if name in ("Blue", "Red") else "black",
                width=8, command=lambda c=bgr: self.set_color(c)
            ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            self.root, text="Eraser", width=12, bg="#dddddd",
            command=self.set_eraser
        ).pack(**pad)

        ttk.Label(self.root, text="Brush Size").pack(**pad)
        self.brush_slider = tk.Scale(
            self.root, from_=2, to=40, orient=tk.HORIZONTAL,
            command=lambda v: setattr(self, "brush_size", int(v))
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(fill=tk.X, padx=20)

        ttk.Separator(self.root).pack(fill=tk.X, pady=10)

        ttk.Label(self.root, text="HSV Calibration (marker color)",
                  font=("Segoe UI", 10, "bold")).pack(**pad)

        self.hsv_sliders = {}
        labels = ["H min", "S min", "V min", "H max", "S max", "V max"]
        defaults = DEFAULT_HSV_LOWER + DEFAULT_HSV_UPPER
        maxvals = [179, 255, 255, 179, 255, 255]
        for label, default, mx in zip(labels, defaults, maxvals):
            row = tk.Frame(self.root)
            row.pack(fill=tk.X, padx=15)
            tk.Label(row, text=label, width=6, anchor="w").pack(side=tk.LEFT)
            s = tk.Scale(row, from_=0, to=mx, orient=tk.HORIZONTAL)
            s.set(default)
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.hsv_sliders[label] = s

        ttk.Separator(self.root).pack(fill=tk.X, pady=10)

        recog_row = tk.Frame(self.root)
        recog_row.pack(**pad)
        tk.Button(recog_row, text="Recognize Letter", bg="#5bc0de", fg="white",
                  command=self.recognize_letter).pack(side=tk.LEFT, padx=5)
        self.recognized_var = tk.StringVar(value="Draw a letter, then click Recognize")
        ttk.Label(self.root, textvariable=self.recognized_var,
                  font=("Segoe UI", 12, "bold")).pack(**pad)

        ttk.Separator(self.root).pack(fill=tk.X, pady=10)

        btn_row = tk.Frame(self.root)
        btn_row.pack(**pad)
        tk.Button(btn_row, text="Clear Canvas", bg="#f0ad4e",
                  command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_row, text="Quit", bg="#d9534f", fg="white",
                  command=self.quit).pack(side=tk.LEFT, padx=5)

    def set_color(self, bgr):
        self.draw_color = bgr
        self.eraser_on = False

    def set_eraser(self):
        self.eraser_on = True

    def clear_canvas(self):
        self.canvas[:] = 0
        self.points.clear()

    def recognize_letter(self):
        if self.letter_model is None:
            self.recognized_var.set(
                "Model not found - run train_letter_classifier.py first"
            )
            return

        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        points = cv2.findNonZero(gray)
        if points is None:
            self.recognized_var.set("Canvas is empty - draw a letter first")
            return

        # crop to the drawn strokes' bounding box
        x, y, w, h = cv2.boundingRect(points)
        crop = gray[y:y + h, x:x + w]

        # pad to a square (centered) so the letter isn't stretched on resize
        side = max(w, h)
        pad_top = (side - h) // 2
        pad_bottom = side - h - pad_top
        pad_left = (side - w) // 2
        pad_right = side - w - pad_left
        squared = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

        # extra margin around the letter (EMNIST characters aren't edge-to-edge)
        margin = max(4, side // 6)
        squared = cv2.copyMakeBorder(
            squared, margin, margin, margin, margin,
            cv2.BORDER_CONSTANT, value=0
        )

        # smooth + resize to EMNIST's 28x28 format, then flatten & normalize
        squared = cv2.GaussianBlur(squared, (5, 5), 0)
        resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_CUBIC)
        flat = (resized.reshape(1, 784) / 255.0)

        pred = self.letter_model.predict(flat)[0]   # EMNIST 'letters': 1=A ... 26=Z
        letter = chr(pred + 64)                      # 1->'A', 2->'B', ...
        self.recognized_var.set(f"Recognized: {letter}")

    def get_hsv_bounds(self):
        s = self.hsv_sliders
        lower = np.array([s["H min"].get(), s["S min"].get(), s["V min"].get()])
        upper = np.array([s["H max"].get(), s["S max"].get(), s["V max"].get()])
        return lower, upper

    # ------------------------------------------------------------------
    # Core video / tracking loop, driven by tkinter's event loop
    # ------------------------------------------------------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)  # mirror for natural movement
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = self.get_hsv_bounds()
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > MIN_CONTOUR_AREA:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

        # update stroke trail: None marks a break (pen "lifted")
        self.points.append(center)

        # draw connected strokes onto the persistent canvas
        color = (0, 0, 0) if self.eraser_on else self.draw_color
        thickness = self.brush_size * 3 if self.eraser_on else self.brush_size
        pts = self.points
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(self.canvas, pts[i - 1], pts[i], color, thickness)

        # composite canvas over live frame (only where something's drawn)
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
        inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
        frame_bg = cv2.bitwise_and(frame, inv_mask)
        combined = cv2.add(frame_bg, self.canvas)

        status = f"Color: {'Eraser' if self.eraser_on else self._color_name()}  Brush: {self.brush_size}"
        cv2.putText(combined, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        cv2.imshow("Air Canvas - Camera", combined)
        cv2.imshow("Marker Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self.clear_canvas()
        elif key == ord('q'):
            self.quit()
            return

        self.root.after(10, self.update_frame)

    def _color_name(self):
        for name, bgr in DRAW_COLORS.items():
            if bgr == self.draw_color:
                return name
        return "Custom"

    def quit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        self.root.after(0, self.update_frame)
        self.root.mainloop()


if __name__ == "__main__":
    app = AirCanvas()
    app.run()
