#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CB513_URL="https://www.compbio.dundee.ac.uk/jpred4/downloads/513_distribute.tar.gz"
EXPECTED_SHA256="6621aa7a36a45b9cf505e00dab1d41a7fe848ba61d43d35986e502b4063683e2"
EXPECTED_ALL_COUNT=513

BASE_DIR="${REPO_ROOT}/data/raw/cb513"
ARCHIVE_NAME="513_distribute.tar.gz"
ARCHIVE_PATH="${BASE_DIR}/${ARCHIVE_NAME}"
EXTRACT_DIR="${BASE_DIR}/513_distribute"
SHA256_FILE="${BASE_DIR}/SHA256SUMS"
MANIFEST_FILE="${BASE_DIR}/MANIFEST.txt"

VERIFY_ONLY=0
FORCE=0
SKIP_LOCK=0

usage() {
  cat <<'EOF'
Usage: ./scripts/download_cb513.sh [OPTIONS]

Options:
  --verify-only   Verify existing local archive and extracted files only (no download, no writes)
  --force         Re-download archive and rebuild extracted directory
  --skip-lock     Do not apply read-only permissions at the end
  -h, --help      Show this help message
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmds() {
  local cmd
  for cmd in bash curl tar sha256sum find sort wc awk diff mktemp chmod date; do
    command -v "${cmd}" >/dev/null 2>&1 || die "Missing required command: ${cmd}"
  done
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --verify-only)
        VERIFY_ONLY=1
        ;;
      --force)
        FORCE=1
        ;;
      --skip-lock)
        SKIP_LOCK=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
    shift
  done
}

unlock_for_force_if_needed() {
  if [[ "${FORCE}" -eq 1 && -d "${BASE_DIR}" ]]; then
    chmod -R u+w "${BASE_DIR}" || true
  fi
}

download_archive() {
  mkdir -p "${BASE_DIR}"
  local tmp_archive
  tmp_archive="$(mktemp "${BASE_DIR}/.${ARCHIVE_NAME}.tmp.XXXXXX")"
  echo "Downloading CB513 archive from official JPred source..."
  if ! curl -L --retry 5 --retry-delay 2 --connect-timeout 20 \
    -o "${tmp_archive}" "${CB513_URL}"; then
    rm -f "${tmp_archive}"
    die "Failed to download ${CB513_URL}. Check network and retry."
  fi
  mv "${tmp_archive}" "${ARCHIVE_PATH}"
}

verify_archive_hash() {
  [[ -f "${ARCHIVE_PATH}" ]] || die "Archive not found: ${ARCHIVE_PATH}"
  echo "${EXPECTED_SHA256}  ${ARCHIVE_PATH}" | sha256sum -c - >/dev/null
}

extract_archive_atomically() {
  local extract_tmp
  local staged_extract
  local backup_extract

  extract_tmp="$(mktemp -d "${BASE_DIR}/.extract_tmp.XXXXXX")"
  staged_extract="${BASE_DIR}/.staged_513_distribute.$$"
  backup_extract="${BASE_DIR}/.backup_513_distribute.$$"
  trap 'rm -rf "${extract_tmp}" "${staged_extract}" "${backup_extract}"' ERR

  tar -xzf "${ARCHIVE_PATH}" -C "${extract_tmp}"
  [[ -d "${extract_tmp}/513_distribute" ]] || die "Archive extracted but 513_distribute/ is missing."

  mv "${extract_tmp}/513_distribute" "${staged_extract}"

  if [[ -d "${EXTRACT_DIR}" ]]; then
    mv "${EXTRACT_DIR}" "${backup_extract}"
  fi
  mv "${staged_extract}" "${EXTRACT_DIR}"
  rm -rf "${backup_extract}"
  rm -rf "${extract_tmp}"
  trap - ERR
}

collect_all_files() {
  [[ -d "${EXTRACT_DIR}" ]] || die "Extracted directory not found: ${EXTRACT_DIR}"
  find "${EXTRACT_DIR}" -type f -name '*.all' -print | LC_ALL=C sort
}

validate_all_count() {
  local count
  count="$(collect_all_files | wc -l | tr -d ' ')"
  [[ "${count}" -eq "${EXPECTED_ALL_COUNT}" ]] || die "Expected ${EXPECTED_ALL_COUNT} .all files, found ${count}."
}

generate_current_checksums() {
  local out_file="$1"
  (
    cd "${BASE_DIR}"
    sha256sum "${ARCHIVE_NAME}"
    find "513_distribute" -type f -name '*.all' -print | LC_ALL=C sort | while IFS= read -r rel_file; do
      sha256sum "${rel_file}"
    done
  ) > "${out_file}"
}

write_checksums_file() {
  generate_current_checksums "${SHA256_FILE}"
}

verify_checksums_file() {
  [[ -f "${SHA256_FILE}" ]] || die "Missing checksum file: ${SHA256_FILE}"
  local current_tmp
  current_tmp="$(mktemp)"
  generate_current_checksums "${current_tmp}"
  if ! diff -u "${SHA256_FILE}" "${current_tmp}" >/dev/null; then
    rm -f "${current_tmp}"
    die "Checksum verification against ${SHA256_FILE} failed."
  fi
  rm -f "${current_tmp}"
}

write_manifest() {
  mapfile -t rel_files < <(collect_all_files | sed "s#^${EXTRACT_DIR}/##")
  local file_count
  local now_utc
  local actual_sha
  local i
  file_count="${#rel_files[@]}"
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  actual_sha="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"

  {
    echo "CB513 Raw Dataset Manifest"
    echo "Generated UTC: ${now_utc}"
    echo "Source URL: ${CB513_URL}"
    echo "Archive: ${ARCHIVE_NAME}"
    echo "Archive SHA256 (expected): ${EXPECTED_SHA256}"
    echo "Archive SHA256 (actual): ${actual_sha}"
    echo ".all file count: ${file_count}"
    echo
    echo "First 10 files (sorted):"
    for (( i = 0; i < 10 && i < file_count; i++ )); do
      echo "  - ${rel_files[$i]}"
    done
    echo
    echo "Last 10 files (sorted):"
    local start=0
    if (( file_count > 10 )); then
      start=$((file_count - 10))
    fi
    for (( i = start; i < file_count; i++ )); do
      echo "  - ${rel_files[$i]}"
    done
  } > "${MANIFEST_FILE}"
}

lock_raw_data() {
  chmod -R a-w "${BASE_DIR}"
  find "${BASE_DIR}" -type d -exec chmod a+rx {} +
}

print_summary() {
  local count
  local actual_sha
  count="$(collect_all_files | wc -l | tr -d ' ')"
  actual_sha="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"

  echo
  echo "CB513 dataset ready."
  echo "  Base directory : ${BASE_DIR}"
  echo "  Archive path   : ${ARCHIVE_PATH}"
  echo "  Extracted path : ${EXTRACT_DIR}"
  echo "  SHA256 file    : ${SHA256_FILE}"
  echo "  Manifest file  : ${MANIFEST_FILE}"
  echo "  Archive SHA256 : ${actual_sha}"
  echo "  .all count     : ${count}"
}

main() {
  require_cmds
  parse_args "$@"
  unlock_for_force_if_needed
  mkdir -p "${BASE_DIR}"

  if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
    [[ -f "${ARCHIVE_PATH}" ]] || die "Verify-only requested but archive is missing: ${ARCHIVE_PATH}"
    [[ -d "${EXTRACT_DIR}" ]] || die "Verify-only requested but extracted data is missing: ${EXTRACT_DIR}"
    verify_archive_hash
    validate_all_count
    verify_checksums_file
    print_summary
    exit 0
  fi

  if [[ "${FORCE}" -eq 1 || ! -f "${ARCHIVE_PATH}" ]]; then
    download_archive
  else
    echo "Archive already exists, skipping download: ${ARCHIVE_PATH}"
  fi

  verify_archive_hash

  if [[ "${FORCE}" -eq 1 || ! -d "${EXTRACT_DIR}" ]]; then
    extract_archive_atomically
  else
    echo "Extracted directory already exists, skipping extraction: ${EXTRACT_DIR}"
  fi

  validate_all_count

  if [[ "${FORCE}" -eq 1 || ! -f "${SHA256_FILE}" || ! -f "${MANIFEST_FILE}" ]]; then
    write_checksums_file
    write_manifest
  fi

  verify_checksums_file

  if [[ "${SKIP_LOCK}" -eq 0 ]]; then
    lock_raw_data
  fi

  print_summary
}

main "$@"
