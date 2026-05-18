#!/bin/bash

set -e

REQUIRED_PYTHON="3.10"
VECTOR_DB="src/ingest.py"
TARGET_FILE="src/app.py"

# Load dependencies from requirements.txt
REQUIREMENTS_FILE="requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "requirements.txt not found."
    exit 1
fi

PYTHON_CMD="python$REQUIRED_PYTHON"

# Check Python
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "$PYTHON_CMD not found."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install "python@$REQUIRED_PYTHON"
        else
            echo "Homebrew not found. Install Python $REQUIRED_PYTHON manually."
            exit 1
        fi

    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt &> /dev/null; then
            sudo apt update
            sudo apt install -y \
              "python$REQUIRED_PYTHON" \
              "python$REQUIRED_PYTHON-dev" \
              "python$REQUIRED_PYTHON-distutils"
        else
            echo "Unsupported Linux package manager."
            exit 1
        fi
    else
        echo "Unsupported OS."
        exit 1
    fi
fi

PYTHON_PATH=$(command -v "$PYTHON_CMD")

echo "Using Python:"
"$PYTHON_PATH" --version

# Check pip
if ! "$PYTHON_PATH" -m pip --version &> /dev/null; then
    echo "pip not found for $PYTHON_CMD."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        "$PYTHON_PATH" -m ensurepip --upgrade
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt install -y "python$REQUIRED_PYTHON-pip"
    fi
fi

# Install dependencies using the same Python
echo "Installing dependencies from $REQUIREMENTS_FILE..."
"$PYTHON_PATH" -m pip install --upgrade pip
"$PYTHON_PATH" -m pip install -r "$REQUIREMENTS_FILE"

# Run target file using same Python
if [ ! -f "$TARGET_FILE" ]; then
    echo "Target file not found: $TARGET_FILE"
    exit 1
fi

echo "Running $TARGET_FILE with $PYTHON_CMD..."
"$PYTHON_PATH" "$VECTOR_DB"
"$PYTHON_PATH" -m streamlit run "$TARGET_FILE"
