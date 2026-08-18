#!/bin/bash
# Build the preserved PENGWIN 2026 Task 2 v3.7 container image.
#
# Build context is the repository root (one level up from scripts/) so the
# Dockerfile can COPY both inference/ (Task 2 entrypoint + vendored Task 1
# pipeline) and code_task1/ (trainer-discovery shim source).
#
# 모델 가중치는 이미지에 포함하지 않는다 — Grand Challenge Models 탭에 model.tar.gz 로
# 따로 올리고 런타임에 /opt/ml/model 로 해제된다(Task 1 과 동일 tarball 재사용).
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-pengwin-task2-interact:latest}"

cd "$(dirname "$0")/.."
docker build \
    -t "$IMAGE_TAG" \
    .

echo
echo "Built image: $IMAGE_TAG"
docker images "$IMAGE_TAG" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedAt}}"
