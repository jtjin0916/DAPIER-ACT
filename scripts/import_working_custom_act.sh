#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/DAPIER-ACT"
ROBOWIN="${HOME}/RoboTwin"

SRC_TRAIN="${ROBOWIN}/dapier_act"
SRC_EVAL="${ROBOWIN}/XPolicyLab/policy/DAPIER_ACT"

DST_TRAIN="${ROOT}/reference/robotwin_dapier_act"
DST_EVAL="${ROOT}/reference/xpolicylab_dapier_act"

[[ -d "${ROOT}/.git" ]] || { echo "ERROR: ${ROOT} is not a git repository."; exit 1; }
[[ -d "${SRC_TRAIN}" ]] || { echo "ERROR: missing ${SRC_TRAIN}"; exit 1; }
[[ -d "${SRC_EVAL}" ]] || { echo "ERROR: missing ${SRC_EVAL}"; exit 1; }

echo "== Source summary =="
du -sh "${SRC_TRAIN}" "${SRC_EVAL}"

echo
echo "== Copying working Custom ACT references =="

mkdir -p "${ROOT}/reference"
rm -rf "${DST_TRAIN}" "${DST_EVAL}"

rsync -a   --exclude '__pycache__/'   --exclude '*.pyc'   --exclude '*.pt'   --exclude '*.pth'   --exclude '*.ckpt'   --exclude 'checkpoints/'   --exclude 'data/'   "${SRC_TRAIN}/" "${DST_TRAIN}/"

rsync -a   --exclude '__pycache__/'   --exclude '*.pyc'   --exclude '*.pt'   --exclude '*.pth'   --exclude '*.ckpt'   --exclude 'checkpoints/'   --exclude 'data/'   "${SRC_EVAL}/" "${DST_EVAL}/"

echo
echo "== Copied files =="
find "${ROOT}/reference" -type f | sort

echo
echo "STOP BEFORE COMMIT:"
echo "1) inspect copied files for credentials and absolute local paths"
echo "2) preserve upstream copyright/license headers"
echo "3) identify ACT/DP3-derived file origins"
echo "4) XPolicyLab-derived files require Apache-2.0 attribution"
echo "5) run git diff --stat and git diff -- reference/"
