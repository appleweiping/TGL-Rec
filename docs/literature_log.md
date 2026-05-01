# Literature Log

All entries in this batch were checked on 2026-04-29. Summaries are written in
project wording and should be rechecked before final experiments.

## RecBole Benchmark Library

- Source: https://recbole.io/docs/user_guide/model_intro.html
- Code: https://github.com/RUCAIBox/RecBole
- Venue/date: CIKM 2021 library paper; docs current as RecBole 1.2.1.
- Core idea in own words: unified PyTorch recommender benchmark suite with Pop,
  ItemKNN, BPR, LightGCN, GRU4Rec, SASRec, BERT4Rec, SR-GNN, FEARec, and many
  other models under one runner/evaluator.
- License: MIT in official GitHub repository.
- Dataset/protocol: top-N recommendation; supports full-sort evaluation for many models.
- Relevance: must-use integration path candidate for Module 2 baseline reproduction.
- Compute/API needs: CPU feasible for Pop/ItemKNN/BPR on small data; GPU recommended for
  transformer and graph neural baselines.
- Reproduction risks: do not mix RecBole default splits with project temporal splits.
- Data-format follow-up checked 2026-05-01:
  - Atomic files: https://recbole.io/atomic_files.html
  - Data settings / `benchmark_filename`: https://recbole.io/docs/user_guide/config/data_settings.html
  - Evaluation settings / full ranking: https://recbole.io/docs/user_guide/config/evaluation_settings.html
  - Project decision: export project `train`/`val`/`test` labels as RecBole benchmark files for
    general CF baselines first. Sequential RecBole models need a separate history-aware adapter
    before reportable SASRec/BERT4Rec/TiSASRec runs.

## MovieLens-1M Official Data

- Source: https://grouplens.org/datasets/movielens/1m/
- Direct archive: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- README: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
- Venue/date: GroupLens stable benchmark, released 2003.
- Core idea in own words: timestamped movie ratings; useful for first deterministic
  sequential recommendation preprocessing path.
- Code: local `tglrec preprocess movielens-1m`.
- License: research use with acknowledgement under GroupLens terms; no commercial or
  revenue-bearing use without permission.
- Dataset/protocol: `ratings.dat` is `UserID::MovieID::Rating::Timestamp`; this repo writes
  temporal leave-one-out and global-time splits.
- Relevance: required first real dataset for Module 1.
- Compute/API needs: CPU only.
- Reproduction risks: explicit ratings may have weaker next-need structure than ecommerce.

## Amazon Reviews 2023

- Source: https://amazon-reviews-2023.github.io/main.html
- Hugging Face: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Venue/date: 2023 McAuley Lab release; paper "Bridging Language and Items for Retrieval
  and Recommendation".
- Core idea in own words: large ecommerce review corpus with review text, item metadata,
  links, fine-grained timestamps, and standard splits.
- Code/license: dataset access via project site/Hugging Face; license terms must be checked
  before redistribution.
- Dataset/protocol: timestamped reviews and rich item text make it a better next-need test
  than MovieLens.
- Relevance: required next dataset family after MovieLens.
- Compute/API needs: category subsets are CPU-feasible; full corpus is large.
- Reproduction risks: category files are large; choose a bounded subset and log version.

## BPR-MF

- Source: https://arxiv.org/abs/1205.2618
- Venue/date: UAI 2009.
- Core idea in own words: pairwise ranking objective for implicit feedback matrix
  factorization.
- Code/license: use local clean implementation or RecBole BPR under MIT.
- Dataset/protocol: implicit top-N ranking; explicit ratings need a documented threshold or
  all-positive policy.
- Relevance: must-run non-sequential CF baseline.
- Compute/API needs: CPU feasible on ML-1M.
- Reproduction risks: negative sampling and seen-item filtering must match shared evaluator.

## GRU4Rec

- Source/code: https://github.com/hidasib/GRU4Rec
- Venue/date: ICLR 2016 original; CIKM 2018 top-k-gain follow-up.
- Core idea in own words: recurrent session/sequential baseline using GRU state over
  interaction sequences.
- License: check official repository before reuse.
- Dataset/protocol: official repo emphasizes dataset-specific preprocessing and warns that
  preprocessing differences make results incomparable.
- Relevance: must-run sequential baseline.
- Compute/API needs: GPU recommended for real runs; RecBole implementation may be simpler.
- Reproduction risks: session assumptions may not match long user histories without adaptation.

## SASRec

- Source: https://arxiv.org/abs/1808.09781
- Code: https://github.com/kang205/SASRec
- Venue/date: ICDM 2018.
- Core idea in own words: causal self-attention over item sequences for next-item prediction.
- License: Apache-2.0 in official repository.
- Dataset/protocol: timestamp-sorted user-item interactions.
- Relevance: must-run sequential baseline.
- Compute/API needs: GPU recommended; CPU smoke possible.
- Reproduction risks: official code is old TensorFlow/Python 2 style; RecBole may be more
  practical if documented.

## BERT4Rec

- Source: https://arxiv.org/abs/1904.06690
- Code: https://github.com/FeiSun/BERT4Rec
- Replicability study: https://arxiv.org/abs/2207.07483
- Venue/date: CIKM 2019; replicability study RecSys 2022.
- Core idea in own words: bidirectional Transformer trained with masked item prediction for
  sequential recommendation.
- License: Apache-2.0 in official repository.
- Dataset/protocol: sequential recommendation; results are sensitive to training/eval details.
- Relevance: must-run baseline, with reproducibility caution.
- Compute/API needs: GPU recommended.
- Reproduction risks: BERT4Rec-vs-SASRec conclusions vary across publications.

## TiSASRec

- Source: https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf
- DOI page: https://dl.acm.org/doi/10.1145/3336191.3371786
- Code: https://github.com/JiachengLi1995/TiSASRec
- Venue/date: WSDM 2020.
- Core idea in own words: self-attention augmented with time-interval encodings.
- License: no license file found in official repo on 2026-04-29.
- Dataset/protocol: timestamped sequences; repo includes ML-1M example.
- Relevance: must-run time-aware sequential baseline.
- Compute/API needs: GPU recommended.
- Reproduction risks: license unclear; prefer clean reimplementation if code reuse matters.

## LightGCN

- Source: https://arxiv.org/abs/2002.02126
- Code: https://github.com/kuandeng/LightGCN
- Venue/date: SIGIR 2020.
- Core idea in own words: simplified static user-item graph collaborative filtering through
  linear neighborhood propagation.
- License: no license file found in official repo on 2026-04-29; RecBole path is MIT.
- Dataset/protocol: full ranking over unobserved items in official examples.
- Relevance: must-run graph CF baseline.
- Compute/API needs: GPU recommended for full runs; CPU smoke possible.
- Reproduction risks: graph must use training interactions only.

## TGN

- Source/code: https://github.com/twitter-research/tgn
- Venue/date: ICML Workshop / arXiv 2020.
- Core idea in own words: temporal graph memory and message passing for dynamic graph events.
- License: check official repository before reuse.
- Dataset/protocol: dynamic link prediction on event streams.
- Relevance: optional dynamic graph baseline or design reference for Module 7.
- Compute/API needs: GPU recommended.
- Reproduction risks: not a recommender baseline out of the box.

## EvolveGCN

- Source: https://arxiv.org/abs/1902.10191
- Code: https://github.com/IBM/EvolveGCN
- Venue/date: AAAI 2020.
- Core idea in own words: evolves GCN parameters over graph snapshots using recurrent updates.
- License: check official repository before reuse.
- Dataset/protocol: dynamic graph node/link tasks.
- Relevance: optional dynamic graph reference.
- Compute/API needs: GPU recommended.
- Reproduction risks: snapshot formulation may not match event-level recommender data.

## Lost in Sequence / LLM-SRec

- Source: https://arxiv.org/abs/2502.13909
- Code: https://github.com/Sein-Kim/LLM-SRec
- Venue/date: KDD 2025 research track per arXiv metadata.
- Core idea in own words: diagnoses whether LLM recommenders use sequence information and
  proposes distilling CF sequential representations into lightweight LLM-adjacent modules.
- License: no license file found in official repo on 2026-04-29.
- Dataset/protocol: directly aligned with this project's sequence-use hypothesis.
- Relevance: must-run or must-compare LLM4Rec baseline after non-LLM baselines.
- Compute/API needs: GPU likely; API optional depending on model.
- Reproduction risks: license and LLM dependency footprint need review.

## LLaRA

- Source: https://arxiv.org/abs/2312.02445
- Code: https://github.com/ljy0ustc/LLaRA
- Venue/date: SIGIR 2024.
- Core idea in own words: aligns conventional recommender ID embeddings with LLM input space
  through hybrid prompts and curriculum training.
- License: Apache-2.0.
- Dataset/protocol: sequential recommendation with text and ID channels.
- Relevance: optional LLM4Rec baseline.
- Compute/API needs: GPU and LLM fine-tuning.
- Reproduction risks: expensive; defer until Module 2+.

## P5

- Source: https://arxiv.org/abs/2203.13366
- Code: https://github.com/jeykigung/P5
- Venue/date: RecSys 2022.
- Core idea in own words: casts recommendation tasks as language processing with prompts and
  sequence-to-sequence pretraining.
- License: check official repository before reuse.
- Dataset/protocol: multi-task recommendation; not strictly next-item only.
- Relevance: optional LLM4Rec baseline or related work.
- Compute/API needs: GPU for T5-style training.
- Reproduction risks: RecSys 2024 reproducibility discussions warn P5 can be hard to reproduce.

## ReLLa

- Source: https://arxiv.org/abs/2308.11131
- Code: https://github.com/LaVieEnRose365/ReLLa
- Venue/date: 2023 arXiv / LLM-enhanced recommendation line.
- Core idea in own words: retrieval-enhanced LLM framework for long sequential behavior
  comprehension in zero-shot and few-shot recommendation.
- License: check official repository before reuse.
- Dataset/protocol: sequential behavior text prompts with retrieval augmentation.
- Relevance: optional LLM retrieval baseline.
- Compute/API needs: GPU or API/local LLM.
- Reproduction risks: task and compute heavier than Module 0/1.

## G-Refer

- Source: https://arxiv.org/abs/2502.12586
- Code: https://github.com/Yuhan1i/G-Refer
- Venue/date: WWW 2025 Oral per repository README.
- Core idea in own words: retrieves collaborative graph evidence, translates it into text, and
  uses graph retrieval-augmented LLMs for explainable recommendation.
- License: no license file found in official repo on 2026-04-29.
- Dataset/protocol: explanation-oriented graph retrieval and graph translation.
- Relevance: optional static graph-to-language comparator; important related work.
- Compute/API needs: GPU/LLM; API optional for explanation evaluation.
- Reproduction risks: not a temporal next-item method; adapt ideas cleanly.

## G-CRS

- Source: https://arxiv.org/abs/2503.06430
- Venue/date: 2025 arXiv.
- Core idea in own words: graph retrieval-augmented LLM for conversational recommendation.
- Code/license: not verified as official in this pass.
- Dataset/protocol: conversational recommendation.
- Relevance: related work only unless conversational setting is added.
- Compute/API needs: LLM/API or local LLM.
- Reproduction risks: task mismatch.

## FEARec

- Source: https://recbole.io/docs/recbole/recbole.model.sequential_recommender.fearec.html
- Code: https://github.com/sudaada/FEARec
- Venue/date: SIGIR 2023.
- Core idea in own words: frequency-enhanced hybrid attention for sequential recommendation.
- License: MIT if using RecBole; check official repo before reuse.
- Dataset/protocol: RecBole integration available.
- Relevance: optional strong sequential baseline after required baselines.
- Compute/API needs: GPU recommended.
- Reproduction risks: not time-aware in the TiSASRec sense.

## BSARec

- Source: https://arxiv.org/abs/2312.10325
- Code: https://github.com/yehjin-shin/BSARec
- Venue/date: AAAI 2024.
- Core idea in own words: adds frequency-domain inductive bias beyond vanilla self-attention.
- License: check official repository before reuse.
- Dataset/protocol: repo examples include Beauty, Sports, Toys, Yelp, ML-1M, and LastFM.
- Relevance: optional strong recent sequential baseline.
- Compute/API needs: GPU recommended.
- Reproduction risks: extra preprocessing/tuning burden.

## Meta Generative Recommenders / HSTU

- Source/code: https://github.com/meta-recsys/generative-recommenders
- Paper linked from repo: https://arxiv.org/abs/2402.17152
- Venue/date: ICML 2024 line of generative sequential recommendation.
- Core idea in own words: large-scale generative sequential recommender framework including
  HSTU and SASRec-style configs.
- License: check official repository before reuse.
- Dataset/protocol: repo has reproducible configs for industrial-scale sequential models.
- Relevance: related/stretch baseline, not Module 0/1.
- Compute/API needs: heavy GPU likely.
- Reproduction risks: industrial-scale assumptions may distract from temporal diagnostic focus.
