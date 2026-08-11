#!/bin/bash
# Start a virtual X server so the headless Unity player (AnimalAI.x86_64) can
# open a window, and run the command under VirtualGL so that the player gets a
# real OpenGL core profile from the NVIDIA driver instead of Xvfb's software GL.
set -e

if [ ! -e "/tmp/.X11-unix/X${DISPLAY#:}" ]; then
    Xvfb "${DISPLAY}" -screen 0 "${XVFB_SCREEN}" -nolisten tcp &
    for _ in $(seq 50); do
        if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
            break
        fi
        sleep 0.2
    done
    if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
        echo "entrypoint: Xvfb failed to start on ${DISPLAY}" >&2
        exit 1
    fi
fi

# Without a GPU there is nothing for VirtualGL to render on; the command still
# runs, but the Unity environment will not start.
if [ -e /dev/nvidiactl ]; then
    exec vglrun -d "${VGL_DEVICE}" "$@"
fi

echo "entrypoint: no NVIDIA device found, running without VirtualGL" >&2
exec "$@"
