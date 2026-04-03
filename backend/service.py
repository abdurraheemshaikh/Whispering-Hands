from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


LABELS_DICT = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "O",
    13: "P",
    14: "Q",
    15: "R",
    16: "S",
    17: "T",
    18: "U",
    19: "V",
    20: "W",
    21: "X",
    22: "Y",
    23: "Z",
    24: "1",
    25: "2",
    26: "3",
    27: "4",
    28: "5",
    29: "6",
    30: "7",
    31: "8",
    32: "9",
    33: " ",
    34: "DELETE",
}


@dataclass(frozen=True)
class PredictionResult:
    gesture: str | None
    confidence: float | None


def _resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is not None:
        return Path(model_path).resolve()
    return Path(__file__).resolve().parents[1] / "model2.p"


class HandSignPredictor:
    def __init__(self, model_path: str | Path | None = None) -> None:
        resolved_model_path = _resolve_model_path(model_path)
        with resolved_model_path.open("rb") as model_file:
            model_dict: dict[str, Any] = pickle.load(model_file)

        self.model = model_dict["model"]
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.3,
            max_num_hands=1,
        )

    @staticmethod
    def decode_image(image_data: str) -> np.ndarray:
        payload = image_data.strip()
        if "," in payload:
            payload = payload.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except Exception as exc:  # pragma: no cover - defensive input handling
            raise ValueError("Invalid base64 image payload") from exc

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        return frame

    def predict_image(self, image_data: str) -> PredictionResult:
        frame = self.decode_image(image_data)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return PredictionResult(gesture=None, confidence=None)

        hand_landmarks = results.multi_hand_landmarks[0]
        x_values: list[float] = []
        y_values: list[float] = []

        for landmark in hand_landmarks.landmark:
            x_values.append(landmark.x)
            y_values.append(landmark.y)

        min_x = min(x_values)
        min_y = min(y_values)

        data_aux: list[float] = []
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min_x)
            data_aux.append(landmark.y - min_y)

        prediction = self.model.predict([np.asarray(data_aux)])
        probabilities = self.model.predict_proba([np.asarray(data_aux)])

        predicted_index = int(prediction[0])
        gesture = LABELS_DICT.get(predicted_index)
        confidence = float(np.max(probabilities[0])) if len(probabilities) else None

        return PredictionResult(gesture=gesture, confidence=confidence)
