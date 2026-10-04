from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from capstone.commands.executor import CommandExecutor
from capstone.dialog.rules import parse_command
from capstone.vision.face_auth import FaceAuthenticator


def _authenticator(data_dir: Path, threshold: float = 65.0) -> FaceAuthenticator:
    return FaceAuthenticator(
        data_dir / "face_model.yml", data_dir / "face_labels.json", threshold
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capstone", description="Local face-gated rule-based command demo."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    enroll = subparsers.add_parser("enroll", help="Enroll the authorized demo user.")
    enroll.add_argument("--name", required=True, help="Name shown after authentication.")
    enroll.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".capstone"),
        help="Directory for local face data (default: .capstone).",
    )
    enroll.add_argument(
        "--camera", type=int, default=0, help="Webcam index (default: 0)."
    )
    enroll.add_argument(
        "--image",
        type=Path,
        action="append",
        help="Reference image; repeat to provide multiple images.",
    )

    run = subparsers.add_parser("run", help="Authenticate, accept commands, and act.")
    run.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".capstone"),
        help="Directory for local face data (default: .capstone).",
    )
    run.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    run.add_argument(
        "--threshold",
        type=float,
        default=65.0,
        help="LBPH match distance; lower is stricter (default: 65).",
    )
    run.add_argument(
        "--voice",
        action="store_true",
        help="Use optional offline Vosk speech input instead of typing.",
    )
    return parser


def _enroll(args: argparse.Namespace, authenticator: FaceAuthenticator) -> None:
    if args.image:
        count = authenticator.enroll_image_files(args.image, args.name)
    else:
        count = authenticator.enroll_webcam(args.name, args.camera)
    print(f"Enrolled {args.name} with {count} face sample(s) in {args.data_dir}.")


def _run(args: argparse.Namespace, authenticator: FaceAuthenticator) -> int:
    try:
        result = authenticator.authenticate_webcam(args.camera)
    except RuntimeError as exc:
        print(f"Face check error: {exc}")
        return 1
    if not result.authenticated or not result.name:
        print(f"Access denied: {result.reason}")
        return 1

    print(f"Access granted. Welcome, {result.name}.")
    executor = CommandExecutor(result.name)
    while True:
        try:
            if args.voice:
                from capstone.voice.vosk_input import listen_once

                command = listen_once()
                print(f"You said: {command}")
            else:
                command = input("command> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        except RuntimeError as exc:
            print(f"Voice input error: {exc}")
            return 1

        intent = parse_command(command)
        print(executor.execute(intent))
        if intent.name == "exit":
            return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.action == "enroll":
            authenticator = _authenticator(args.data_dir)
            _enroll(args, authenticator)
            return
        authenticator = _authenticator(args.data_dir, args.threshold)
        exit_code = _run(args, authenticator)
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        parser.error(str(exc))
    raise SystemExit(exit_code)
