#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/DAPIER-ACT"
REF="${ROOT}/reference/robotwin_dapier_act"
DST="${ROOT}/dapier_act"
ARCHIVE="${ROOT}/experimental/provisional_model_scaffold"

cd "${ROOT}"

[[ -d .git ]] || { echo "ERROR: ${ROOT} is not a git repository"; exit 1; }

declare -A EXPECTED=(
  [dataset.py]="1f1c54120bfd67698691f4df519cc13a7d946bddc16f64b4fc142a4a0d76df60"
  [dataset_multimodal.py]="a0bb54762f40b1dc27fcfc0c9a25f06ebd67e9b4e91a7dbe9645ef4abfe0cf4c"
  [model.py]="62dca4bc1166d7c1741a86c63fe8b4dc5ead303dc79adf21d335729e5b46c82b"
  [model_spatial.py]="e5ca2dc863bf904b874279688226a5936b1a88af7ab80f05c3defa511b43840d"
  [model_multimodal.py]="215490cdf85f08b18bd24473859ae4a8e50e710910f3ad1796ee71f2164122b0"
  [pointcloud_preprocess.py]="8638ce5905ec4899b8737991bef9cb5ad4419073517e90c2a273b48b25eac34e"
  [train_multimodal.py]="fddc194c8518000091b1d7ac62b2ea42e0e4e56e0ca1fddf92a5d2b892f9dcf3"
)

echo "== 1. Verify active reference hashes =="
for name in "${!EXPECTED[@]}"; do
  src="${REF}/${name}"
  [[ -f "${src}" ]] || { echo "ERROR: missing ${src}"; exit 1; }
  actual="$(sha256sum "${src}" | awk '{print $1}')"
  if [[ "${actual}" != "${EXPECTED[$name]}" ]]; then
    echo "ERROR: ${name} does not match the verified DP3-head source."
    echo " expected ${EXPECTED[$name]}"
    echo " actual   ${actual}"
    exit 1
  fi
  echo "OK  ${name}"
done

echo
echo "== 2. Preserve the provisional public model scaffold =="
if [[ -d "${DST}/model" ]]; then
  rm -rf "${ARCHIVE}"
  mkdir -p "$(dirname "${ARCHIVE}")"
  mv "${DST}/model" "${ARCHIVE}"
  echo "Moved dapier_act/model/ -> experimental/provisional_model_scaffold/"
else
  echo "No dapier_act/model/ directory found; continuing."
fi

echo
echo "== 3. Install the behavior-preserving active runtime =="
for name in \
  dataset.py \
  dataset_multimodal.py \
  model.py \
  model_spatial.py \
  model_multimodal.py \
  pointcloud_preprocess.py \
  train_multimodal.py
do
  cp -a "${REF}/${name}" "${DST}/${name}"
done

find "${DST}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo "== 4. Verify public runtime == reference =="
for name in \
  dataset.py \
  dataset_multimodal.py \
  model.py \
  model_spatial.py \
  model_multimodal.py \
  pointcloud_preprocess.py \
  train_multimodal.py
do
  cmp -s "${REF}/${name}" "${DST}/${name}" || {
    echo "ERROR: public ${name} differs from reference"
    exit 1
  }
  echo "MATCH ${name}"
done

echo
echo "== 5. Compile =="
python -m compileall -q \
  "${DST}/model.py" \
  "${DST}/model_spatial.py" \
  "${DST}/model_multimodal.py" \
  "${DST}/pointcloud_preprocess.py" \
  "${DST}/dataset.py" \
  "${DST}/dataset_multimodal.py" \
  "${DST}/train_multimodal.py"

echo
echo "Integration complete."
echo "Run next:"
echo "  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests"
echo "  git status"
echo "  git diff --stat"
