from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AuthResult:
    authenticated: bool
    name: str | None
    reason: str


class FaceAuthenticator:
    """Detect faces and compare them with a locally stored LBPH face model."""

    def __init__(self, model_path: Path, labels_path: Path, threshold: float = 65.0):
        self.model_path = model_path
        self.labels_path = labels_path
        self.threshold = threshold
        self.detector = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        if self.detector.empty():
            raise RuntimeError("OpenCV's frontal-face detector could not be loaded.")

    @staticmethod
    def _recognizer():
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "OpenCV face recognition is unavailable. Install opencv-contrib-python."
            )
        return cv2.face.LBPHFaceRecognizer_create()

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

    @staticmethod
    def _prepare_face(gray: np.ndarray) -> np.ndarray:
        face = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        return cv2.equalizeHist(face)

    def enroll_images(self, images: list[np.ndarray], name: str) -> int:
        if not name.strip():
            raise ValueError("The enrolled person's name cannot be empty.")

        samples: list[np.ndarray] = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces) != 1:
                raise ValueError(
                    "Each enrollment image must contain exactly one detectable face."
                )
            x, y, width, height = faces[0]
            samples.append(self._prepare_face(gray[y : y + height, x : x + width]))

        if not samples:
            raise ValueError("At least one enrollment image is required.")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        recognizer = self._recognizer()
        recognizer.train(samples, np.ones(len(samples), dtype=np.int32))
        recognizer.write(str(self.model_path))
        self.labels_path.write_text(
            json.dumps({"1": name.strip()}, indent=2), encoding="utf-8"
        )
        return len(samples)

    def enroll_image_files(self, image_paths: list[Path], name: str) -> int:
        images: list[np.ndarray] = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read enrollment image: {image_path}")
            images.append(image)
        return self.enroll_images(images, name)

    def enroll_webcam(
        self, name: str, camera_index: int = 0, sample_count: int = 20
    ) -> int:
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"Could not open camera {camera_index}.")

        samples: list[np.ndarray] = []
        try:
            print("Look at the camera; collecting face samples. Press q to finish.")
            while len(samples) < sample_count:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning frames.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
                )
                for x, y, width, height in faces:
                    cv2.rectangle(
                        frame, (x, y), (x + width, y + height), (0, 200, 0), 2
                    )
                cv2.putText(
                    frame,
                    f"Samples: {len(samples)}/{sample_count}  (q to finish)",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 0),
                    2,
                )
                cv2.imshow("Capstone enrollment", frame)
                if len(faces) == 1 and len(samples) < sample_count:
                    x, y, width, height = faces[0]
                    samples.append(self._prepare_face(gray[y : y + height, x : x + width]))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()

        if not samples:
            raise RuntimeError("No face samples were collected; enrollment was not saved.")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        recognizer = self._recognizer()
        recognizer.train(samples, np.ones(len(samples), dtype=np.int32))
        recognizer.write(str(self.model_path))
        self.labels_path.write_text(
            json.dumps({"1": name.strip()}, indent=2), encoding="utf-8"
        )
        return len(samples)

    def authenticate_webcam(
        self,
        camera_index: int = 0,
        timeout_seconds: int = 20,
        required_matches: int = 3,
    ) -> AuthResult:
        if not self.model_path.is_file() or not self.labels_path.is_file():
            return AuthResult(False, None, "No enrolled face was found. Enroll first.")

        labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        recognizer = self._recognizer()
        recognizer.read(str(self.model_path))
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"Could not open camera {camera_index}.")

        matches = 0
        deadline = time.monotonic() + timeout_seconds
        try:
            print("Face check: look at the camera. Press q to cancel.")
            while time.monotonic() < deadline:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning frames.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
                )
                if len(faces) == 1:
                    x, y, width, height = faces[0]
                    face = self._prepare_face(gray[y : y + height, x : x + width])
                    label, confidence = recognizer.predict(face)
                    matches = matches + 1 if label == 1 and confidence <= self.threshold else 0
                    cv2.rectangle(
                        frame, (x, y), (x + width, y + height), (0, 200, 0), 2
                    )
                else:
                    matches = 0
                cv2.putText(
                    frame,
                    f"Matching frames: {matches}/{required_matches}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 0),
                    2,
                )
                cv2.imshow("Capstone face check", frame)
                if matches >= required_matches:
                    return AuthResult(True, labels.get("1"), "Face matched.")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return AuthResult(False, None, "Face check cancelled.")
        finally:
            camera.release()
            cv2.destroyAllWindows()
        return AuthResult(False, None, "Face did not match before the check timed out.")
