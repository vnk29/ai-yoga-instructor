#!/bin/bash
# Launches Plank Posture Coach with the correct Python installation.
# Run:  bash run.sh
# Or:   chmod +x run.sh && ./run.sh

PYTHON=/usr/local/bin/python3

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Edit PYTHON= in run.sh to point to your Python 3.10+ installation."
    exit 1
fi

# Point Python's SSL stack at the certifi CA bundle so Gradium TTS works on macOS.
CERTIFI_PATH=$("$PYTHON" -c "import certifi; print(certifi.where())" 2>/dev/null)
if [ -n "$CERTIFI_PATH" ]; then
    export SSL_CERT_FILE="$CERTIFI_PATH"
    export REQUESTS_CA_BUNDLE="$CERTIFI_PATH"
fi

cd "$(dirname "$0")"
exec "$PYTHON" main.py "$@"
