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
