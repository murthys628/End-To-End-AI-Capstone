# End-to-End AI Capstone

A small, local-first Python CLI that combines three reusable pieces:

1. **Vision:** OpenCV face detection and local face matching gate access.
2. **Dialog:** a deterministic phrase-to-intent rule set.
3. **Actions:** an allowlisted executor for simple commands (it never runs shell input).

The `src/capstone/vision`, `dialog`, `commands`, and `voice` packages are intentionally
separate so interns can replace one component with code from an earlier project.

## Requirements

- Python 3.10 or newer
- A webcam for the interactive face enrollment and authentication flow
- A working desktop display for OpenCV's camera preview

## Setup

From this folder, create and activate a virtual environment, then install the project:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the demo

Enroll through the webcam (press `q` to finish early):

```powershell
python -m capstone enroll --name "Alex"
```

Or enroll from one or more local reference images:

```powershell
python -m capstone enroll --name "Alex" --image .\samples\alex-1.jpg --image .\samples\alex-2.jpg
```

Authenticate and start the text command loop:

```powershell
python -m capstone run
```

Try `help`, `status`, `time`, `hello`, `say hello capstone`, or `exit`.
Use `python -m capstone run --camera 1` if the webcam is not camera 0, and
pass `--data-dir <path>` after `enroll` or `run` to choose a different local
data directory.

### Optional offline voice input

Install the voice dependencies and download a Vosk model for your language from
[Vosk models](https://alphacephei.com/vosk/models). Set the model directory and run:

```powershell
python -m pip install -e ".[voice]"
$env:CAPSTONE_VOSK_MODEL = "C:\models\vosk-model-small-en-us-0.15"
python -m capstone run --voice
```

Voice capture and transcription run locally. The voice mode captures one five-second
utterance per command. Text mode is the default and requires no speech model.

## Tests

```powershell
python -m pytest
```

## Privacy and limitations

Face model and label files are written under `.capstone/` by default; that directory
is ignored by Git. Enrollment images are read locally and are not copied into the
data directory. Delete `.capstone/` to remove the enrolled model.

This is an educational demo, **not a secure authentication system**. Haar cascades
and LBPH are lightweight classic computer-vision techniques, can produce false
matches or rejections, and do not provide liveness or spoof detection. Do not use
them to protect real accounts, devices, facilities, or sensitive data. Use a
multi-image enrollment set and a controlled lighting/camera setup for a more
repeatable demonstration.
