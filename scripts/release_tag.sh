#!/usr/bin/env bash
set -euo pipefail

# Create and push an annotated git tag for a release.
# Usage:
#   scripts/release_tag.sh                # uses version from pyproject.toml
#   scripts/release_tag.sh 0.3.1.post1    # explicit version

PYPROJECT_FILE="pyproject.toml"
REMOTE="origin"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this script must be run inside a git repository." >&2
  exit 1
fi

if [[ ! -f "${PYPROJECT_FILE}" ]]; then
  echo "Error: ${PYPROJECT_FILE} not found." >&2
  exit 1
fi

if ! git remote get-url "${REMOTE}" >/dev/null 2>&1; then
  echo "Error: git remote '${REMOTE}' does not exist." >&2
  exit 1
fi

VERSION="${1:-}"
if [[ -z "${VERSION}" ]]; then
  VERSION="$(awk -F'"' '/^version = "/ { print $2; exit }' "${PYPROJECT_FILE}")"
fi

if [[ -z "${VERSION}" ]]; then
  echo "Error: could not determine version (pass it as first argument)." >&2
  exit 1
fi

TAG="v${VERSION}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ "${CURRENT_BRANCH}" != "main" ]]; then
  echo "Warning: current branch is '${CURRENT_BRANCH}', not 'main'." >&2
  echo "Aborting to avoid tagging the wrong commit." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean. Commit or stash changes first." >&2
  exit 1
fi

if git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "Error: local tag '${TAG}' already exists." >&2
  exit 1
fi

git fetch --tags "${REMOTE}" >/dev/null
if git ls-remote --tags "${REMOTE}" "refs/tags/${TAG}" | grep -q .; then
  echo "Error: remote tag '${TAG}' already exists on '${REMOTE}'." >&2
  exit 1
fi

echo "Creating annotated tag '${TAG}' on commit $(git rev-parse --short HEAD)"
git tag -a "${TAG}" -m "Release ${VERSION}"

echo "Pushing '${TAG}' to '${REMOTE}'"
git push "${REMOTE}" "${TAG}"

echo "Done. Next step: create/publish a GitHub Release for '${TAG}' to trigger PyPI publishing."
