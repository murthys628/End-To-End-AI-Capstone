# Short demo script

1. Install the package and test dependencies using the commands in [README.md](README.md).
2. Enroll the presenter: `python -m capstone enroll --name "Alex"`.
   Show that the local face model is stored in `.capstone/`.
3. Start the flow: `python -m capstone run`. Show that an unrecognized face or
   cancelled face check does not reach the command prompt.
4. After access is granted, enter `help`, `status`, `say modular components`,
   and `exit`. Point out that text maps to a small intent and only built-in
   actions can run.
5. Optional: enroll with `--image` or repeat commands with offline Vosk voice
   input as documented in [README.md](README.md).
6. Run `python -m pytest` to demonstrate the camera-independent dialog and
   executor checks.

The face check is for demonstration only: it is not liveness detection or a
production security control.
