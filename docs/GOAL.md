# TGL-Rec 项目总目标（GOAL — 长期执行者必读）

> 本文件是项目的标准 goal 注入文本。任何长期执行 agent（codex/GPT-5.5 等）每个 session
> 必须先读本文件 + `CLAUDE.md` + `AGENTS.md` + `docs/HOW_TO_RUN_CC_PACE.md` +
> `docs/codex_project_memory.md`（按此顺序），再行动。

## 一、最终目标（唯一的成功定义）

把 TGL-Rec 推进到**顶会投稿就绪**（SIGIR/WWW/KDD/RecSys/NeurIPS/ICLR 级别），即同时满足：

1. **主表**：CC-PACE 方法在 8 域（beauty/books/electronics/movies/sports/toys/home/tools）
   × 8 个冻结官方 baseline 的同候选协议下完整对比，beauty 必须先达 SOTA
   （beat promax NDCG@10 = 0.1506，paired-bootstrap p<0.05），再 rollout 其他 7 域。
2. **三个必做附加实验**（导师指定，见 `docs/paper_followup_experiments.md`）：
   observation（popularity-collapse 现象，~2 域）、ablation（逐组件，诚实报告无效组件）、
   hyperparameter 扫描（matplotlib 曲线）+ 一张框架总览图。
3. **论文**：基于真实产出的 metrics 写完全文（绝不先写结论），过一轮顶会式内部评审 gate。

## 二、不可违反的纪律（违反 = 工作无效）

- 实验**只在服务器** pony-rec-gpu 上跑；本地只开发/测试/commit；**绝不从服务器 commit/push**。
- 同候选协议冻结：101 候选/用户、Qwen3-8B backbone、HR@5/10/20 + NDCG@5/10/20 + MRR。
  评测任务文件只认 `~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/`
  的导出（baselines 就是在这些 candidate set 上打的分）。
- 8 baseline × 8 域证据已冻结（`data/pony_official_baselines/`），永不重跑。
- 绝不伪造/外推任何实验数字；无显著性检验不下"超过"结论。
- 每完成一步：更新 agentmemory + 项目文档（codex_project_memory/CONTEXT/README）+ commit/push。
- TGL-Rec 与 Pony / TRUCE-Rec 完全独立：可读共享协议数据，绝不混方法代码。
- beauty 不 SOTA 就不准 rollout；卡住时走三席 ARIS 讨论重设计（CLAUDE.md 规则 9）。

## 三、当前状态快照（2026-06-13 凌晨）

- 方法 CC-PACE 已实现并 CPU 测试（`src/llm4rec/methods/cc_pace/`，33 个单测绿）。
- 数据接缝已接通：CF artifacts（frozen SASRec）+ profile slots，服务器已产出
  （`outputs/cc_pace_beauty/cf_artifacts.json` + `profiles.json`）。
- **zero-shot kill test 进行中**：`text_only` 变体 973 用户完成，NDCG@10 = 0.1108；
  `full` 变体断点续跑中（checkpoint `outputs/cc_pace_beauty/full.json.per_user.jsonl`）。
  服务器队列 `~/projects/gpu_queue3.sh`（log 同名 .log）自动跑完 full 并产出
  `outputs/cc_pace_beauty/go_verdict.json`。
- 分支：`feat/cc-pace-data-seams`（GitHub 已推送；服务器同分支检出）。

## 四、决策树（严格按此执行）

1. **等 `go_verdict.json`**：
   - `decision == "GO"`（full NDCG@10 ≥ 0.13 且 full > text_only 0.1108）→ 步骤 2。
   - `KILL_OR_REFRAME` → 三席 ARIS 讨论（Opus-lead + Opus#2 + GPT-5.5 xhigh，
     ≥8/10 设计 gate）重设计方法，重跑 beauty zero-shot，直到 GO。**不准带病训练 LoRA**。
2. **LoRA 训练**：`scripts/train_cc_pace_lora.py`（已就绪，含防泄漏设计与 fold A/B）。
   训练后用 `scripts/cc_pace_beauty.py --adapter <lora_dir>` 评估全部 5 个变体
   （full/text_only/no_residualizer/no_shrinkage/rich_residualizer）。
3. **STRONG GO 判定**（全部满足才算 beauty SOTA）：post-LoRA NDCG@10 ≥ 0.1506
   （paired-bootstrap p<0.05，用 per_user.jsonl）、panel-corruption 降 ≥30%、
   text_only 低于 CF baseline（机制证明）。需补写 panel-corruption 检查脚本。
   正式分数按 `source_event_id,user_id,item_id,score` schema 导出存档。
4. **8 域 rollout**：同一方法/超参，逐域串行（磁盘只剩 ~17GB，先清旧 artifacts）。
   每域：CF artifacts → profiles → zero-shot → LoRA → 评估，对比该域 baseline 行。
5. **三个附加实验 + 总览图**（主表完成后才开始）→ 填表 → 写论文 → 内部评审 gate →
   告知用户投稿就绪。

## 五、运维要点（实操踩坑已验证）

- 服务器：`ssh pony-rec-gpu`，python = `~/miniconda3/envs/tglrec-lora/bin/python`
  （文档写的 tglrec 环境不存在）。RTX 4090 实际 47.4GB。
- **服务器不通 GitHub**：同步 = 本地 `git bundle create` → scp 到 `~/projects/TGL-Rec/.git/`
  → 服务器 `git fetch <bundle> +分支:refs/remotes/bundle/x && git reset --hard bundle/x`。
- GPU 串行纪律：同一时间只跑一个大任务；启动前查 `nvidia-smi` 空闲 ≥40GB；
  与 TRUCE-Rec 项目共享这块卡，互相排队。
- 长任务必须 `setsid nohup` + 落盘 checkpoint；驱动脚本已支持 per-user 续跑。
- pgrep 守护进程时注意自匹配死锁（模式写成 `[c]c_pace...` 或用 marker 文件）。
- transformers 5.7：`DynamicCache.batch_repeat_interleave` 就地返回 None；
  cache 走 `cache.layers[i].keys/values`；批量扩 KV cache 必须小批量（B=8）。

## 六、汇报格式（每个复杂任务结束时）

what changed / what was tested / complete or blocked / next concrete plan /
current gate toward submission-ready。
