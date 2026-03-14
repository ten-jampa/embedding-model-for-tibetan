#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/embedding-tibetan-env/bin/python"

SRC_A="/Users/ten-jampa/Documents/personal_projects/embedding-model-for-tibetan-rkts-pilot/smoke_output/data/converted/unicode/BonBkz/001.txt"
SRC_B="/Users/ten-jampa/Documents/personal_projects/embedding-model-for-tibetan-rkts-pilot/smoke_output/data/converted/unicode/BonBkz/050.txt"

INPUT_DIR="${REPO_ROOT}/output/pairwise_smoke/inputs"
OUT_DIR="${REPO_ROOT}/output/pairwise_smoke/botok_gemma_cpu_short"
TEXT_A="${INPUT_DIR}/rkts_bonbkz_001_1p5k.txt"
TEXT_B="${INPUT_DIR}/rkts_bonbkz_050_1p5k.txt"

mkdir -p "${INPUT_DIR}" "${OUT_DIR}"

echo "[1/3] Preparing tiny input slices (1500 chars each)..."
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

src_a = Path("/Users/ten-jampa/Documents/personal_projects/embedding-model-for-tibetan-rkts-pilot/smoke_output/data/converted/unicode/BonBkz/001.txt")
src_b = Path("/Users/ten-jampa/Documents/personal_projects/embedding-model-for-tibetan-rkts-pilot/smoke_output/data/converted/unicode/BonBkz/050.txt")
out_dir = Path("output/pairwise_smoke/inputs")
out_dir.mkdir(parents=True, exist_ok=True)

chunk = 1500
text_a = src_a.read_text(encoding="utf-8")[:chunk]
text_b = src_b.read_text(encoding="utf-8")[:chunk]

path_a = out_dir / "rkts_bonbkz_001_1p5k.txt"
path_b = out_dir / "rkts_bonbkz_050_1p5k.txt"
path_a.write_text(text_a, encoding="utf-8")
path_b.write_text(text_b, encoding="utf-8")

print(f"text_a={path_a} chars={len(text_a)}")
print(f"text_b={path_b} chars={len(text_b)}")
PY

echo "[2/3] Running pairwise (Gemma Mitra, CPU)..."
cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/run_pairwise_text_similarity.py \
  --text-a "${TEXT_A}" \
  --text-b "${TEXT_B}" \
  --output-dir "${OUT_DIR}" \
  --input-format unicode \
  --botok-cache-dir .cache/botok/dialect_packs \
  --top-k 10 \
  --batch-size 1 \
  --device cpu

echo "[3/3] Done. Artifacts:"
echo "  ${OUT_DIR}/run_manifest.json"
echo "  ${OUT_DIR}/topk_pairs.csv"
echo "  ${OUT_DIR}/topk_pairs.jsonl"

