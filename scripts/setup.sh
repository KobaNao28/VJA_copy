#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv || true
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

if command -v apt-get >/dev/null 2>&1; then
  echo "Tesseract OCR(多言語)のインストールには sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn を実行してください"
fi

echo "セットアップ完了。'source .venv/bin/activate' で仮想環境を有効化してください。"
