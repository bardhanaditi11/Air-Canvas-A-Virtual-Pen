import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def catch_all_errors(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "trace": traceback.format_exc().splitlines()[-5:],  # last 5 lines
        },
    )

@app.get("/api/health")
def health():
    return {"status": "ok"}


import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

TRAIN_CSV = "emnist-letters-train.csv"
TEST_CSV = "emnist-letters-test.csv"
MODEL_PATH = "letter_model.joblib"

# Keep training time reasonable on a laptop CPU by using a solid subset
# of the ~124,800 available training rows. Raise this (up to ~124800)
# for higher accuracy at the cost of a longer training time.
SUBSET = 40000


def load_csv(path):
    """
    Load one of Kaggle's crawford/emnist CSVs.
    Format: first column = label (1-26, A-Z), remaining 784 columns =
    pixel values for a 28x28 image, but stored transposed/rotated
    relative to how the letter actually looks - so we correct the
    orientation the same way the dataset's example notebooks do.
    """
    df = pd.read_csv(path, header=None)
    labels = df.iloc[:, 0].to_numpy()
    images = df.iloc[:, 1:].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)

    # Fix EMNIST's stored orientation (flip, then rotate) for every image
    images = np.flip(images, axis=2)
    images = np.rot90(images, k=1, axes=(1, 2))

    images = images.reshape(-1, 784) / 255.0
    return images, labels


def main():
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
        print(f"Missing {TRAIN_CSV} and/or {TEST_CSV}.")
        print("Download them from https://www.kaggle.com/datasets/crawford/emnist")
        print("and place both files in this same folder, then re-run this script.")
        return

    print("Loading training data...")
    X_train, y_train = load_csv(TRAIN_CSV)
    print("Loading test data...")
    X_test, y_test = load_csv(TEST_CSV)

    X_train, y_train = X_train[:SUBSET], y_train[:SUBSET]

    print(f"Training MLPClassifier on {len(X_train)} samples...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=40,
        alpha=1e-4,
        solver="adam",
        random_state=1,
        verbose=True,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc * 100:.2f}%")

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    print("You can now use the 'Recognize Letter' button in air_canvas.py")


if __name__ == "__main__":
    main()
