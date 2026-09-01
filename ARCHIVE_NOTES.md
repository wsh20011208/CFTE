# Archive cleanup applied

This archive-ready copy was made from the final uploaded source snapshots.

Only packaging/reproducibility cleanup was applied:

1. Removed nested `.git`, `__pycache__`, `*.pyc`, local log files and `testing_data/`.
2. Preserved the four scientific project source trees.
3. Added a `vox-256.yaml` gap-3 compatibility alias while keeping the authoritative gap3/gap5/gap7 YAML files.
4. Replaced benchmark wrapper scripts with gap-specific wrappers:
   - correct gap-specific triplet generator,
   - correct gap-specific YAML,
   - corresponding `checkpoint_d3/d5/d7` epoch-199 checkpoint lookup.
5. Removed stale temporary ZIP revision names from benchmark compatibility notes.
6. Renamed the physical-bitstream scripts from uploaded `(2)` filenames to their canonical import names.
7. Updated physical encode/decode config lookup to the gap-3 YAML and made the orchestrator import sibling stage scripts from `physical_bitstream/`.

No model architecture, loss function, benchmark metric implementation, or reported numerical result was changed by this archive cleanup.
