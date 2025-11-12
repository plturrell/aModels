#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="${ROOT_DIR}/checkpoints/archive"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
if [[ ! -d "${ARCHIVE_DIR}" ]]; then
  echo "Archive directory not found: ${ARCHIVE_DIR}" >&2
  exit 0
fi
find "${ARCHIVE_DIR}" -type f -mtime +"${RETENTION_DAYS}" -print -delete
find "${ARCHIVE_DIR}" -type d -empty -print -delete
