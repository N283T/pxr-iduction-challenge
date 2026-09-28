# Model artifacts

Model artifacts smaller than GitHub's 100 MB per-file limit are committed to
this repository, including the corresponding files under
`track1_activity/checkpoints/`.

Twenty-four larger checkpoint files (7,695,165,552 bytes in total) are stored
in the `pxr-data-v1.0.0` GitHub Release as a split, uncompressed tar archive:

`pxr-large-model-artifacts-v1.0.0.tar.segment-000` through
`pxr-large-model-artifacts-v1.0.0.tar.segment-004`.

## Restore the large files

Run from the repository root:

```bash
gh release download pxr-data-v1.0.0 \
  --repo N283T/pxr-iduction-challenge \
  --pattern 'pxr-large-model-artifacts-v1.0.0.tar.segment-*'

sha256sum -c models/LARGE_MODEL_ARCHIVE_SHA256SUMS

cat pxr-large-model-artifacts-v1.0.0.tar.segment-* | tar -xf -
```

`models/LARGE_MODEL_ARTIFACTS_SHA256SUMS` contains checksums for the 24
restored source files. `models/LARGE_MODEL_ARCHIVE_SHA256SUMS` contains
checksums for the five downloadable archive segments.
