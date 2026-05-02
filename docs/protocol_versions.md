# Protocol Versions

Phase 8 introduces `protocol_v1` as the planned paper-scale protocol version.

The dry-run freeze command is:

```bash
python scripts/freeze_protocol.py --version protocol_v1 --dry-run
```

The command saves:

- `outputs/launch/paper_v1/protocol/protocol_manifest.json`
- `outputs/launch/paper_v1/protocol/frozen_split_manifest.json`
- `outputs/launch/paper_v1/protocol/frozen_candidate_manifest.json`

The protocol manifest records split strategy, candidate protocol, metric set, method set, seed set,
dataset config hashes, experiment config hashes, and the current code commit.

Phase 8 defaults to metadata-only freeze. Full split and candidate materialization is allowed only
with an explicit `--materialize` flag in a later launch step after dataset readiness is `READY`.
Existing non-dry-run protocol manifests are not overwritten without `--force-new-version`.
