# nanochat 初始阅读 TODO

## 阅读顺序
- [ ] `README.md`：了解项目目标、完整流水线与产出。
- [ ] `speedrun.sh`：快速掌握端到端脚本各阶段任务。
- [ ] `scripts/base_train.py`：梳理训练超参、配置覆盖与主循环。
- [ ] `nanochat/gpt.py`：深入模型结构、Block 设计与优化器分组。
- [ ] `nanochat/engine.py`：理解 KV Cache 维护、采样逻辑与推理流程。

## 核心模块
- [ ] `nanochat/dataset.py`：熟悉数据分片下载、Parquet 迭代与训练/验证划分。
- [ ] `nanochat/dataloader.py`：掌握分布式 Token 流式加载、BOS 拼接与 GPU 异步搬运。
- [ ] `nanochat/tokenizer.py`：了解 GPT-4 风格分词器、RustBPE+tiktoken 结合与特殊符号。
- [ ] `nanochat/configurator.py`：学习命令行覆盖配置的 Poor Man’s Configurator。
- [ ] `nanochat/common.py`：认识日志、DDP 初始化及基础目录工具函数。

## 下一步
- [ ] 运行 `python -m pytest tests/test_rustbpe.py -v -s`，验证分词实现并建立测试基线。
- [ ] 阅读 `scripts/chat_web.py`，掌握 FastAPI 推理服务启动流程。
- [ ] 浏览 `nanochat/checkpoint_manager.py`，了解模型保存与加载接口。
- [ ] 查看 `nanochat/report.py`，理解报告生成逻辑以扩展自定义指标。
