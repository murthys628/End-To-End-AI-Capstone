import sys
from pathlib import Path

import numpy as np

from capstone.commands.executor import CommandExecutor
from capstone.dialog.rules import Intent, parse_command
from capstone import cli
from capstone.vision.face_auth import FaceAuthenticator


class FixedFaceDetector:
    def detectMultiScale(self, image, **kwargs):
        return [(0, 0, image.shape[1], image.shape[0])]


def test_parse_command_normalizes_spacing_and_case():
    assert parse_command("  WHAT time is it ") == Intent("time")


def test_say_command_preserves_original_argument():
    assert parse_command("say Hello, World!") == Intent("say", "Hello, World!")


def test_unknown_command_is_not_executed_as_a_shell_command():
    response = CommandExecutor("Ari").execute(parse_command("python something.py"))
    assert response.startswith("I did not recognize")


def test_exit_intent_is_allowlisted():
    assert parse_command("exit") == Intent("exit")
    assert CommandExecutor("Ari").execute(Intent("exit")) == "Goodbye."


def test_voice_option_is_available_after_run_subcommand():
    args = cli._build_parser().parse_args(["run", "--voice", "--camera", "1"])
    assert args.voice is True
    assert args.camera == 1


def test_enroll_dispatch_does_not_require_run_threshold(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["capstone", "enroll", "--name", "Alex"])
    monkeypatch.setattr(
        cli, "_authenticator", lambda data_dir: calls.append(("auth", data_dir)) or object()
    )
    monkeypatch.setattr(
        cli, "_enroll", lambda args, authenticator: calls.append(("enroll", args.name))
    )

    cli.main()

    assert calls == [
        ("auth", Path(".capstone")),
        ("enroll", "Alex"),
    ]


def test_authentication_is_denied_before_camera_when_not_enrolled(tmp_path):
    authenticator = FaceAuthenticator(
        tmp_path / "face_model.yml", tmp_path / "face_labels.json"
    )
    result = authenticator.authenticate_webcam()
    assert result.authenticated is False
    assert result.name is None
    assert "Enroll first" in result.reason


def test_enrollment_writes_local_model_and_label(tmp_path):
    authenticator = FaceAuthenticator(
        tmp_path / "face_model.yml", tmp_path / "face_labels.json"
    )
    authenticator.detector = FixedFaceDetector()
    image = np.zeros((160, 160), dtype="uint8")
    image[20:140, 30:130] = 180

    sample_count = authenticator.enroll_images([image], "Ari")

    assert sample_count == 1
    assert (tmp_path / "face_model.yml").is_file()
    assert '"1": "Ari"' in (tmp_path / "face_labels.json").read_text(encoding="utf-8")
