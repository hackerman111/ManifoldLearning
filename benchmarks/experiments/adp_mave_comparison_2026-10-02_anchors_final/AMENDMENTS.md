# Artifact amendments

- `protocol.json`: the initial run captured only stderr from `Rscript --version`,
  while this host emits the version on stdout. The field was corrected from the
  same configured executable to `Rscript (R) version 4.6.1 (2026-06-24)`.
  Benchmark fits and measured rows were not changed.
