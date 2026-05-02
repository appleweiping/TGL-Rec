# Paper Table Plan

Phase 8 defines table shells before any results exist. Tables must be generated from metrics files;
manual metric values are not allowed.

Planned table shells:

- main accuracy table;
- ablation table;
- long-tail table;
- cold-start table;
- efficiency table;
- diagnostic table.

Each shell specifies input metrics globs, grouping keys, metric columns, significance marker policy,
and CSV/TEX output paths. The table plan is written to:

```text
outputs/launch/paper_v1/table_plan.json
```

The table plan contains no numbers and records
`NO_EXPERIMENTS_EXECUTED_IN_PHASE_8 = true`.
