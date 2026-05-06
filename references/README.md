# Reference Material Policy

This directory is for local research references. Large reference files are not
committed.

## recprefer.zip

The uploaded `recprefer.zip` package may contain NH/NR recommendation papers and
related PDFs. It can be stored locally as:

```text
references/recprefer.zip
```

or extracted locally under:

```text
references/recprefer/
```

PDFs, zip files, and extracted large archives are ignored by git and must not be
committed.

## What May Be Committed Later

Future lightweight notes may be committed, for example:

- `docs/related_work/recprefer_index.md`
- `docs/related_work/baseline_notes.md`
- small manually written summaries with citations and no large copied text.

## How The Papers Should Be Used

The NH/NR reference set should inform:

- baseline families relevant to recommendation experiments;
- NH metrics such as NDCG and Hit Ratio;
- NR metrics such as NDCG and Recall;
- common preprocessing settings;
- minimal-change baselines useful for reviewer-proofing.

Reference notes must distinguish paper claims from Storyflow/TRUCE-Rec results.
This repository must not report an experimental conclusion unless it was
produced by this project's code and logs.
