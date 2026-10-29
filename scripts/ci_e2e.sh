#!/usr/bin/env bash
# End-to-end test of the DVC data-versioning workflow, safe to run in CI.
#
# Copies the project into a throwaway git+DVC repo, generates synthetic
# Kaggle-shaped data (the real dataset is not in GitHub), then runs the same
# steps you run by hand: release v1 -> v2 -> v3, compare, time-travel to v1.
# Your real repo, tags and DVC remote are never touched.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
REMOTE="$(mktemp -d)"   # stands in for the DVC remote; kept outside the repo

cd "$PROJECT"
cp -R src scripts tests params.yaml data_versions.yaml dvc.yaml .gitignore "$WORK"/
cd "$WORK"

git init -q -b main
git config user.email "ci@example.com"
git config user.name "ci"
dvc init -q
dvc remote add -d storage "$REMOTE" -q

echo "== 1. Create the full synthetic source dataset and track it"
mkdir -p data/source data/raw reports
python scripts/synthetic_data.py data/source/train_full.csv
dvc add data/source/train_full.csv
git add -A
git commit -qm "scaffold"

echo "== 2. Release v1, v2, v3"
for v in v1 v2 v3; do
  echo "---- $v"
  bash scripts/release_data_version.sh "$v"
done

echo "== 3. Check tags, status and remote"
test "$(git tag | grep -c '^data-v')" -eq 3 || { echo "FAIL: expected 3 data tags"; exit 1; }
dvc status | grep -q "up to date" || { echo "FAIL: dvc status not clean"; dvc status; exit 1; }
test "$(find "$REMOTE" -type f | wc -l)" -gt 0 || { echo "FAIL: nothing pushed to remote"; exit 1; }

echo "== 4. Compare versions"
mkdir -p "$PROJECT/ci-output"
dvc metrics diff data-v1 data-v3 | tee "$PROJECT/ci-output/metrics_diff.txt"
python - <<'PY'
import json, subprocess

def rows(tag):
    out = subprocess.check_output(["git", "show", f"{tag}:reports/data_stats.json"])
    return json.loads(out)["rows"]

r1, r2, r3 = rows("data-v1"), rows("data-v2"), rows("data-v3")
print("rows per version:", r1, r2, r3)
assert 0 < r1 < r2 < r3, "dataset versions must keep growing"
PY

echo "== 5. Time travel back to v1 and reproduce it"
V1_ROWS="$(git show data-v1:reports/data_stats.json | python -c 'import json,sys; print(json.load(sys.stdin)["rows"])')"
git checkout -q -f data-v1
dvc checkout data/raw/train.csv.dvc
dvc repro
test "$(( $(wc -l < data/raw/train.csv) - 1 ))" -eq "$V1_ROWS" \
  || { echo "FAIL: v1 data was not restored"; exit 1; }
test -z "$(git status --porcelain)" \
  || { echo "FAIL: v1 did not reproduce cleanly"; git status --short; exit 1; }

echo "ALL CHECKS PASSED"
