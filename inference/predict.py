import cv2
import os
import numpy as np
import tensorflow as tf


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/lipread_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# VOCABULARY — EXACTLY LIKE ORIGINAL NOTEBOOK
# ============================================================

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(
    vocabulary=vocab,
    oov_token=""
)

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    oov_token="",
    invert=True
)


# ============================================================
# VIDEO PREPROCESSING
# EXACTLY LIKE ORIGINAL NOTEBOOK
# ============================================================

def load_video(path):

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    if not os.path.isabs(path):
        path = os.path.join(project_root, path)

    path = os.path.normpath(path)

    print("Opening:", path)

    cap = cv2.VideoCapture(path)

    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):

        ret, frame = cap.read()

        if not ret:
            continue

        # IMPORTANT:
        # Original notebook uses RGB grayscale
        frame = tf.image.rgb_to_grayscale(frame)

        # Exact crop from notebook
        frame = frame[190:236, 80:220, :]

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("Could not read any frames from video.")

    # Stack frames
    frames = tf.stack(frames)

    # Exact normalization from notebook
    mean = tf.math.reduce_mean(frames)

    std = tf.math.reduce_std(
        tf.cast(frames, tf.float32)
    )

    frames = tf.cast(
        (frames - mean),
        tf.float32
    ) / std

    return frames


# ============================================================
# PREDICTION
# ============================================================

def predict_video(video_path):

    frames = load_video(video_path)

    print("Video shape:", frames.shape)

    frames = tf.expand_dims(frames, axis=0)

    print("Model input:", frames.shape)

    yhat = model.predict(
        frames,
        verbose=0
    )

    print("Prediction shape:", yhat.shape)

    # CTC decoding
    decoded = tf.keras.backend.ctc_decode(
        yhat,
        input_length=[75],
        greedy=True
    )[0][0].numpy()

    # Convert numbers -> characters
    prediction = tf.strings.reduce_join(
        [num_to_char(x) for x in decoded[0]]
    ).numpy().decode("utf-8")

    # Simple confidence from predicted sequence
    valid_ids = decoded[0][decoded[0] != -1]

    if len(valid_ids) > 0:
        confidences = []

        for i, token_id in enumerate(valid_ids):
            if i < yhat.shape[1]:
                confidences.append(
                    float(np.max(yhat[0, i]))
                )

        confidence = (
            np.mean(confidences) * 100
            if confidences
            else 0.0
        )
    else:
        confidence = 0.0

    return prediction, confidence


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    video_path = input("Enter video path: ")

    prediction, confidence = predict_video(video_path)

    print("\n==============================")
    print("PREDICTION:")
    print(prediction)
    print(f"CONFIDENCE: {confidence:.2f}%")
    print("==============================")