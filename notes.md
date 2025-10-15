# speedrun.sh 串联流程笔记

## 总览
- speedrun.sh 负责端到端跑通 nanochat：从环境准备、分词器训练到多阶段模型训练与报告生成。
- 每个阶段产出的模型或报告被下游脚本使用，形成自洽流水线。

## ASCII 流程图
```
                                      ┌────────────────────────────┐
                                      │   HF FineWeb 数据集 (远程) │
                                      └──────────────┬─────────────┘
                                                     │
speedrun.sh                                          │ downloads
├─ [环境准备] uv venv / uv sync / activate           ▼
│     ↪ 输出：.venv、锁定依赖、~/.cache/nanochat  ┌──────────────────────────────┐
│                                                  │ python -m nanochat.dataset │
├─ [报告初始化] python -m nanochat.report reset     │   (-n 8, -n 240 &)         │
│     ↪ 输出：~/.cache/nanochat/report/*           └──────────────┬─────────────┘
│                                                                │
├─ [分词器阶段]                                          base_data/*.parquet
│   ├─ curl rustup / source cargo env                       │
│   ├─ uv run maturin develop (构建 rustbpe)                │
│   ├─ scripts.tok_train --max_chars=2e9 -------------------┤
│   │     ↪ 输入：base_data 前 8 个 shard                   │
│   │     ↪ 输出：~/.cache/nanochat/tokenizer/{tokenizer.json,…}
│   └─ scripts.tok_eval ------------------------------------┘
│         ↪ 输出：~/.cache/nanochat/report/tokenizer.md
│
├─ [基础预训练] ────────┐
│   ├─ curl eval_bundle.zip → ~/.cache/nanochat/eval_bundle/
│   ├─ torchrun -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
│   │     ↪ 输入：tokenizer + base_data + eval_bundle
│   │     ↪ 输出：
│   │         • ~/.cache/nanochat/tokenized_data/*
│   │         • ~/.cache/nanochat/checkpoints/base/**
│   │         • 报告分段 (loss/日志)
│   │         • 模型构造：`GPT(GPTConfig)` → ModuleDict{wte, h=[Block×depth], lm_head}
│   │         • Block = norm → CausalSelfAttention(旋转位置编码 + QK norm + MQA) → MLP(ReLU²)
│   ├─ torchrun -m scripts.base_loss
│   │     ↪ 输入：base checkpoint + base_data
│   │     ↪ 输出：report/base_loss.md
│   └─ torchrun -m scripts.base_eval
│         ↪ 输入：base checkpoint + eval_bundle
│         ↪ 输出：report/base_eval.md (CORE 指标)
│
├─ [Mid 训练] ─────────┐
│   ├─ torchrun -m scripts.mid_train -- --run=$WANDB_RUN
│   │     ↪ 输入：base checkpoint
│   │     ↪ 输出：checkpoints/mid/**，report/mid_train.md
│   └─ torchrun -m scripts.chat_eval -i mid
│         ↪ 输出：report/chat_eval_mid.md
│
├─ [监督微调 SFT] ─────┐
│   ├─ torchrun -m scripts.chat_sft -- --run=$WANDB_RUN
│   │     ↪ 输入：mid checkpoint
│   │     ↪ 输出：checkpoints/sft/**，report/chat_sft.md
│   └─ torchrun -m scripts.chat_eval -i sft
│         ↪ 输出：report/chat_eval_sft.md
│
├─ [(可选) RL 阶段] ───┐
│   ├─ torchrun -m scripts.chat_rl -- --run=$WANDB_RUN      (默认注释)
│   │     ↪ 输入：sft checkpoint
│   │     ↪ 输出：checkpoints/rl/**
│   └─ torchrun -m scripts.chat_eval -i rl -a GSM8K         (默认注释)
│         ↪ 输出：report/chat_eval_rl.md
│
└─ [生成总报告] python -m nanochat.report generate
      ↪ 汇总 ~/.cache/nanochat/report/* → 生成 report.md（拷贝到仓库根目录）
```

## 关键要素归纳
- **环境与依赖**：uv 管理虚拟环境，所有中间产物统一放在 `~/.cache/nanochat`，便于复用和清理。
- **数据流动**：`nanochat.dataset` 下载 base_data；`scripts.tok_train` 利用前 8 个 shard 训练词表，下游训练复用 tokenizer；主训练阶段需要 eval_bundle 才能进行 CORE 评估。
- **模型阶段**：base → mid → sft → (rl) 每一步的 checkpoint 都是下一步输入，`chat_eval` 在各阶段生成性能报告。
- **报告体系**：`nanochat.report` 先重置报告目录，各阶段脚本写入 Markdown 分段，最后 `report generate` 汇总成最终 `report.md`。

## 核心脚本
- `scripts/base_train.py`：承担基础预训练主循环，解析/覆写超参，调用 `compute_init` 完成 CUDA 与 DDP 初始化，根据 tokenizer 推导模型结构；在 meta 设备构建 `nanochat.gpt.GPT`，迁移到 GPU 后执行 `init_weights()` 并 `torch.compile`。调用 `model.setup_optimizers` 将线性层交给 Muon、embedding/lm_head 使用 AdamW，并开启分布式数据加载。训练过程中按间隔评估验证损失、CORE 指标、采样输出并保存 checkpoint。
- `scripts/base_loss.py`：加载 base checkpoint，对更大规模的 train/val token 流做精细化损失统计，输出 `report/base_loss.md` 作为基础阶段报告。
- `scripts/base_eval.py`：加载 base checkpoint 与 eval_bundle，执行 CORE 任务评估，生成 `report/base_eval.md`。

## base_train.py 预训练动作
- 通过 `get_tokenizer()` 获取词表大小，并按 depth 派生 `model_dim`、`num_heads`、`num_kv_heads`。
- 计算梯度累积步数，确保 `device_batch_size * seq_len * world_size * grad_accum_steps == total_batch_size`。
- 借助 `model.setup_optimizers` 初始化 Muon + AdamW 参数组，并为每个组记录 `initial_lr`。
- 使用 `tokenizing_distributed_data_loader` 构建分布式流式数据加载器，预抓一批数据触发异步读取。
- 定义学习率与 Muon 动量调度函数，在训练循环中按步数动态调整。
- 主循环内执行混合精度前向、loss 计算、反向传播、梯度裁剪、优化器 step/zero_grad，并维持 EMA 损失与训练时间统计。
- 按 `eval_every`、`core_metric_every`、`sample_every` 触发 `evaluate_bpb`、`evaluate_model`、采样生成，同时通过 wandb 或 DummyWandb 记录指标并调用 `save_checkpoint`。
- 循环结束后调用 `compute_cleanup()` 与 `wandb.finish()` 释放分布式资源。
### base_train.py 功能框图
+------------------------------------------------------------+
| Start / User Config                                         |
| - default globals (run, depth, lr, etc.)                    |
| - CLI overrides via nanochat/configurator.py                |
| - user_config = {key: value} snapshot                       |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| compute_init()                                               |
| -> ddp, ddp_rank/local_rank/world_size                       |
| -> device (cuda), master_process flag                        |
| -> autocast_ctx (bfloat16 cuda)                              |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| wandb setup                                                   |
| - use DummyWandb if run=="dummy" or not master               |
| - otherwise wandb.init(project="nanochat", name=run, config) |
| -> wandb_run handle                                           |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| Tokenizer + vocab data                                       |
| - tokenizer = get_tokenizer()                                |
| - token_bytes tensor (device=GPU)                            |
| - vocab_size = tokenizer.get_vocab_size()                    |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| Model config derivation                                      |
| - num_layers = depth                                         |
| - model_dim = depth * 64                                     |
| - num_heads, num_kv_heads ceiling derived                    |
| - model_config_kwargs dict                                   |
| - with torch.device("meta"): GPTConfig + GPT meta model      |
| - model.to_empty("cuda"); model.init_weights()               |
| - orig_model saved; model = torch.compile(...)               |
| - num_params; num_flops_per_token estimate                   |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| Training horizon selection                                   |
| - assert horizon (num_iterations / target_flops / ratio)    |
| - possibly compute num_iterations                            |
| - total_tokens = total_batch_size * num_iterations           |
| - derived ratios & total flops                               |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| Optimizer initialization                                     |
| - optimizers = model.setup_optimizers(...)                   |
| - adamw_optimizer, muon_optimizer handles                    |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| Data loading                                                 |
| - base_dir = get_base_dir(); tokens_dir path                 |
| - train_loader = tokenizing_distributed_data_loader(...,train)|
| - build_val_loader lambda for val split                     |
| - prefetch (x, y) first batch                                |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| Scheduler helpers                                            |
| - get_lr_multiplier(it) with warmup/warmdown                 |
| - get_muon_momentum(it) linear ramp                          |
+------------------------------+-----------------------------+
                               |
                               v
           +--------------------------------------------------------------+
           | Main training loop: for step in range(num_iterations + 1)    |
           | State tracked:                                               |
           |   - min_val_bpb, smooth_train_loss, total_training_time      |
           |   - flops_so_far = num_flops_per_token * total_batch_size * step |
           +----------------------------+-------------------------------+
                                        |
                                        v
      +-------------------------------------------------------------------+
      | [Conditional] Validation                                           |
      | - if last_step or step % eval_every == 0                           |
      | - val_loader = build_val_loader(); eval_steps derived              |
      | - evaluate_bpb(model, val_loader, eval_steps, token_bytes)         |
      | - min_val_bpb update; wandb_run.log(val metrics)                   |
      +-------------------------------------------------------------------+
                                        |
                                        v
      +-------------------------------------------------------------------+
      | [Conditional] CORE metric                                          |
      | - if last_step or (step>0 and step % core_metric_every==0)         |
      | - use orig_model (uncompiled) + evaluate_model(...)                |
      | - results dict {core_metric, centered_results}                     |
      | - wandb_run.log(core metric outputs)                               |
      +-------------------------------------------------------------------+
                                        |
                                        v
      +-------------------------------------------------------------------+
      | [Conditional] Sampling (master only)                               |
      | - prompts list                                                     |
      | - engine = Engine(model, tokenizer)                                |
      | - for each prompt: tokenize + engine.generate_batch(...)           |
      | - decode samples via tokenizer                                     |
      +-------------------------------------------------------------------+
                                        |
                                        v
      +-------------------------------------------------------------------+
      | [Conditional] Checkpointing (master, last_step)                    |
      | - checkpoint_dir under base_dir/base_checkpoints                   |
      | - save_checkpoint(..., orig_model.state_dict(), optimizer states,  |
      |   metadata dict incl. configs and hyperparams)                     |
      +-------------------------------------------------------------------+
                                        |
                                        v
      +-------------------------------------------------------------------+
      | Training step (skipped when last_step)                             |
      | - torch.cuda.synchronize(); t0 timestamp                           |
      | - gradient accumulation for grad_accum_steps:                      |
      |     * with autocast_ctx: loss = model(x,y);                        |
      |     * train_loss = loss.detach(); loss /= grad_accum_steps;        |
      |     * loss.backward(); x,y = next(train_loader) prefetch          |
      | - optional grad_clip on orig_model params                          |
      | - Update LR for each optimizer param group via get_lr_multiplier   |
      | - Update Muon momentum via get_muon_momentum                       |
      | - opt.step() for each optimizer; model.zero_grad(set_to_none=True) |
      | - torch.cuda.synchronize(); dt = t1 - t0                           |
      +-------------------------------------------------------------------+
                                        |
                                        v
      +-------------------------------------------------------------------+
      | Logging & timing                                                   |
      | - smooth_train_loss EMA & debias                                   |
      | - pct_done, tok_per_sec, flops_per_sec, mfu                        |
      | - total_training_time accum (after step>10)                        |
      | - print0(...) step summary                                         |
      | - every 100 steps: wandb_run.log training metrics                  |
      +-------------------------------------------------------------------+
                                        |
                                        v
           +----------------------------+-------------------------------+
           | Loop break when last_step handled (after checkpoint/logs)  |
           +------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------+
| Final reporting                                             |
| - print peak memory, total time, min val bpb                |
| - get_report().log(...) with:                               |
|   * user_config                                             |
|   * training setup stats (params, flops, iterations, etc.)  |
|   * outcomes (val_bpb, core metric, MFU, total flops/time)  |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
| Cleanup                                                     |
| - wandb_run.finish()                                        |
| - compute_cleanup()                                         |
+------------------------------------------------------------+

#### 流程说明
- **用户配置**（`scripts/base_train.py:29-57`）：设定默认超参并允许通过 `nanochat/configurator.py` 覆盖，形成 `user_config` 供日志与报告引用。
- **计算环境初始化**（`scripts/base_train.py:59-63`）：`compute_init()` 建立 DDP/设备上下文，判定 `master_process`，创建 BF16 CUDA 的 `autocast_ctx`。
- **Wandb 会话**（`scripts/base_train.py:65-66`）：决定使用真实 wandb 还是 `DummyWandb`，保证后续日志调用一致。
- **分词器与词表**（`scripts/base_train.py:68-72`）：加载 tokenizer、GPU 上的 `token_bytes`，取得 `vocab_size` 供模型与采样使用。

#### 模型与数据准备
- **模型结构派生**（`scripts/base_train.py:74-125`）：按深度推导层数/维度/头数，在 `meta` 设备上构造 `GPT` 并迁移至 GPU，保留 `orig_model` 及 `torch.compile` 版本，同时统计参数量与单 token FLOPs。
- **训练跨度决策**（`scripts/base_train.py:127-149`）：根据迭代数、目标 FLOPs 或数据:参数比计算 `num_iterations`、`total_tokens` 等核心指标。
- **优化器初始化**（`scripts/base_train.py:129-131`）：调用 `model.setup_optimizers` 返回 AdamW 与 Muon，便于分别调节学习率和动量。
- **数据加载器**（`scripts/base_train.py:133-138`）：定位数据目录，构建训练/验证的分布式加载器并预取首批 `(x, y)`。
- **调度函数**（`scripts/base_train.py:141-163`）：定义学习率倍率与 Muon 动量曲线，为训练步骤更新提供工具。

#### 训练循环关键环节
- **循环控制**（`scripts/base_train.py:166-175`）：遍历 `num_iterations + 1` 步，实时维护最优验证损失、EMA 损失、累计 FLOPs 与耗时。
- **验证评估**（`scripts/base_train.py:176-192`）：按 `eval_every` 或最后一步调用 `evaluate_bpb`，记录最优 bpb，并通过 wandb 上报。
- **CORE 指标**（`scripts/base_train.py:194-206`）：以未编译模型执行 `evaluate_model`，获得 `core_metric` 与 `centered_results`。
- **采样输出**（`scripts/base_train.py:209-228`）：主进程周期性用 `Engine` 生成若干提示的样例，快速感知模型表现。
- **检查点写入**（`scripts/base_train.py:230-247`）：在最后一步由主进程调用 `save_checkpoint`，保存模型、优化器状态与配置元数据。
- **单步训练**（`scripts/base_train.py:253-280`）：混合精度前向、梯度累积、裁剪、调度更新与优化器 `step`，随后清零梯度并统计时延。
- **日志统计**（`scripts/base_train.py:283-304`）：更新 EMA 损失、吞吐量、MFU 等指标，定期打印并写入 wandb。

#### 收尾阶段
- **终端统计**（`scripts/base_train.py:306-309`）：输出峰值显存、总训练时长、最优验证损失。
- **训练报告**（`scripts/base_train.py:311-335`）：通过 `get_report().log` 提交配置、训练设置与结果三大块信息。
- **资源清理**（`scripts/base_train.py:337-339`）：结束 wandb 会话并执行 `compute_cleanup()` 释放分布式资源。

### GPT Block 框图
```
tokens (B,T)
   │
   ▼
Embedding  wte (bf16)
   │
   ▼
rms_norm
   │
   ▼
┌──────────────────────────────────────────────┐
│ Block × n_layer                              │
│                                              │
│  residual +                                  │
│  ┌──────────────────────────────────────────┐ │
│  │ CausalSelfAttention                       │ │
│  │   norm(x)                                 │ │
│  │      ├─ Linear q,k,v (no bias)            │ │
│  │      ├─ rotary cos/sin + QK norm          │ │
│  │      ├─ head transpose + KV cache insert  │ │
│  │      ├─ repeat_kv → MQA                   │ │
│  │      └─ scaled dot-product attention      │ │
│  │            → concat heads → c_proj        │ │
│  └──────────────────────────────────────────┘ │
│                                              │
│  residual +                                  │
│  ┌──────────────────────────────────────────┐ │
│  │ MLP                                       │ │
│  │   norm(x)                                 │ │
│  │     ├─ c_fc (4× width, no bias)           │ │
│  │     ├─ ReLU → square                      │ │
│  │     └─ c_proj (back to width, no bias)    │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
   │
   ▼
rms_norm
   │
   ▼
lm_head (untied, bf16→fp32 logits, softcap)
```

## 训练循环与模型内部的关系
- `scripts/base_train.py` 的训练循环负责“如何训练”：按步骤拉数据、调用 `model(x, y)` 得到 loss、执行 `loss.backward()`、优化器 `step`、梯度裁剪、日志和 checkpoint 等（参见 `scripts/base_train.py:166-304`）。
- `GPT(model_config)` 则定义“模型是什么”：注意力、MLP、残差、LayerNorm 等全部封装在 `nanochat/gpt.py` 的模块实现中。当循环调用 `model(x, y)` 时，内部会通过这些 Transformer block 计算输出，并在反向传播时产生梯度。
- 因此，训练循环与 Transformer block 是上下两层：循环驱动训练流程，模型内部结构在每次 `model(x, y)` 和随后的参数更新中被使用和修改。循环不会并行运行其他模型，只是对同一个 `GPT` 实例反复执行前向/反向。
