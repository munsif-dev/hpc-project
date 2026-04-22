# CB513 Data Provenance

## 1) Dataset identity and purpose
- Dataset: `CB513` (protein secondary-structure prediction benchmark).
- Purpose in this project: raw input source for preprocessing, model training, and Q3 evaluation aligned with the roadmap and paper-inspired setup.

## 2) Primary source and policy
- Primary source URL: `https://www.compbio.dundee.ac.uk/jpred4/downloads/513_distribute.tar.gz`
- Access date: `February 9, 2026`
- Source policy: **official JPred only** for this phase (no mirror fallback in download script).

## 3) Downloaded artifact metadata
- Archive name: `513_distribute.tar.gz`
- HTTP status: `200 OK`
- HTTP `Last-Modified`: `Thu, 06 Mar 2025 16:37:13 GMT`
- Archive size: `1,704,104 bytes`
- Archive SHA256:
  - Expected: `6621aa7a36a45b9cf505e00dab1d41a7fe848ba61d43d35986e502b4063683e2`
  - Actual: `6621aa7a36a45b9cf505e00dab1d41a7fe848ba61d43d35986e502b4063683e2`

## 4) Local storage paths
- Base raw directory: `data/raw/cb513`
- Archive path: `data/raw/cb513/513_distribute.tar.gz`
- Extracted files path: `data/raw/cb513/513_distribute/`
- Checksums file: `data/raw/cb513/SHA256SUMS`
- Manifest file: `data/raw/cb513/MANIFEST.txt`

## 5) Validation results
- Verified archive hash: **PASS**
- Verified extracted `.all` files: `513` files (**PASS**)
- Checksum file consistency (`SHA256SUMS` vs current files): **PASS**
- Latest manifest generation timestamp (UTC): `2026-02-09T13:42:51Z`
- Raw data read-only lock status:
  - `data/raw/cb513`: `dr-xr-xr-x`
  - `data/raw/cb513/513_distribute`: `dr-xr-xr-x`

## 6) Reproduction commands
Run from repo root:

```bash
# Initial acquisition / setup
./scripts/download_cb513.sh

# Verification only (no writes)
./scripts/download_cb513.sh --verify-only

# Force refresh and rebuild
./scripts/download_cb513.sh --force
```

Quick independent checks:

```bash
sha256sum data/raw/cb513/513_distribute.tar.gz
find data/raw/cb513/513_distribute -type f -name '*.all' | wc -l
```

## 7) Script interface used
Implemented at `scripts/download_cb513.sh`:
- `./scripts/download_cb513.sh`
- `./scripts/download_cb513.sh --verify-only`
- `./scripts/download_cb513.sh --force`
- `./scripts/download_cb513.sh --skip-lock`

