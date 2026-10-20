# Lean Critic Lab：DeepSeek 模型替换实验

[English](README.en.md) · [详细实验协议](docs/protocol.md) · [结果与指标](docs/results.md)

在 **CriticLeanBench 公开的 500 条样本及原始评审提示上，只把评审模型替换为 `deepseek-flash`**。每题一次请求，直接采用模型返回的 `Correct` / `Incorrect`。程序负责发送、记录、提取标签和评分，不检查理由是否成立，不根据引文、疑点或规则改写判断。

当前版本为 **0.2.1**。这是一个用途固定的实验工程，不含训练、多模型接入、Agent、回译、多轮复查或人工裁决流程。当前版本不提供自定义 Lean 候选的入口。它评测自然语言数学陈述与候选 Lean 陈述的对应关系；没有运行 Lean，也没有验证完整证明。候选中的 `sorry` 不会被程序判错。

本项目使用作者**发布数据内的原始 `messages`**。作者完整解码参数、运行环境及评分脚本未全部公开，因此我们能固定公开输入和任务，不能声称逐项复现论文私有实验条件。历史实验文件与当前新协议的结果须分开保存。

## 直接查看已完成的实验结果

**只看结果无需安装 Python、下载数据、配置 API 密钥或重新运行实验。** 克隆仓库后，用浏览器打开：

```text
reports/deepseek-bench500-combined-v1-v2/report.html
```

```bash
git clone https://github.com/gjc2/CriticLean.git
cd CriticLean
# macOS；Windows 或 Linux 可在文件管理器中用浏览器打开上述文件
open reports/deepseek-bench500-combined-v1-v2/report.html
```

| 入口 | 内容 |
| --- | --- |
| [合并分析 HTML](reports/deepseek-bench500-combined-v1-v2/report.html) | 500 条总体指标、两阶段实验说明、79 条标签分歧的搜索与展开详情 |
| [原始 500 条 HTML](reports/deepseek-bench500-v1/analysis/report.html) | 8192 输出预算下的原始分析和技术失败记录 |
| [全部 500 条逐题对照](reports/deepseek-bench500-combined-v1-v2/combined/per-case.csv) | 主合并结果的公开标签、模型标签和技术状态 |
| [离线重解析后的逐题对照](reports/deepseek-bench500-combined-v1-v2/combined-reparsed/per-case.csv) | 单独恢复一条已有显式标签的派生结果 |

HTML 的样式、脚本和正文均已内嵌，可离线浏览、搜索和展开详情。`runs/` 与 `reports/` 的原始记录和附件也已提交；保留 clone 后的目录结构，报告中的相对链接即可正常打开。只复制一个 HTML 文件会遗漏链接指向的 CSV、JSON 和原始 trace。GitHub 的 HTML 文件页面显示源码；查看排版后的报告请打开本地文件。

原始 500 条为 **389/500（77.8%）**；保留其中 448 条、用 52 条截断样本的补跑结果替换后为 **419/500（83.8%）**；另列的离线重解析版本为 **420/500（84.0%）**。两阶段共 552 次请求，不含连接测试；并非全部 500 条都在新预算下重新运行。HTML 展开详情覆盖 79 条标签分歧，全部 500 条的结果在逐题 CSV 中，原始输入和回复保存在 `runs/`。

## 实验中什么保持一致

| 项目 | 本实验处理 |
| --- | --- |
| 数据 | 固定版本的 CriticLeanBench，共 500 条 |
| 数学原文和 Lean | 原文保留，包括空白及 Lean 注释 |
| 提示 | 每条数据自带的单个 `user` 消息，逐字发送；两个公开模板分别覆盖 300 / 200 条 |
| 输出任务 | 作者提示要求的 `reasons` 和 `is_assistant_correct: "Correct" / "Incorrect"` |
| 参考标签 | 作者发布的 `tag`，只在本地评分使用，不发给模型 |
| 判定次数 | 每题一次；不重试、不投票、不复查 |
| 唯一模型 | `deepseek-flash` |
| 唯一接口 | `https://api.deepseek.com/responses` |
| 新增评审指令 | 无；不加 system prompt、回译提示或 API JSON schema |
| 程序是否做语义裁决 | 否；只提取显式模型标签 |

DeepSeek 的思考、输出上限和超时是本地运行参数，不能认为它们等同于作者的解码设置。完整边界见[协议](docs/protocol.md)。

## 1. 安装

需要 Python 3.11+。核心调用使用标准库，无需安装 OpenAI SDK；从官方 Parquet 数据转换时需要可选依赖。以下命令在本项目根目录执行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[datasets]'
lean-critic --version
```

已有虚拟环境时可以直接激活并重新安装：`python -m pip install --upgrade --force-reinstall .`，然后确认 `lean-critic --version` 为 `0.2.1`。这里使用普通安装，不使用开发模式安装。修改源码后需要重新执行 `python -m pip install '.[datasets]'`，才能让已安装命令使用新代码。

Windows 的激活命令为 `.venv\Scripts\activate`；下面的密钥输入示例针对 macOS 默认的 zsh。

## 2. 准备固定版本的数据

```bash
lean-critic fetch-criticlean --out data/criticlean-raw

lean-critic prepare-critic \
  --input data/criticlean-raw/raw.jsonl \
  --out data/criticlean-deepseek
```

已有 `data/criticlean-raw/raw.jsonl` 时跳过下载；已有同协议的准备目录时也不必重复准备。准备操作要求新目录或空目录，不会覆盖已有实验输入。

下载命令绑定数据版本和校验和。若联网下载不可用，可以先自行取得对应 Parquet，再使用：

```bash
lean-critic fetch-criticlean \
  --local-parquet /absolute/path/to/test-00000-of-00001.parquet \
  --out data/criticlean-raw
```

准备目录包含：

| 文件 | 用途 |
| --- | --- |
| `cases.jsonl` | 数学原文和候选 Lean，不含参考标签 |
| `author-prompts.jsonl` | 逐字保留的公开提示，与题目内容的哈希绑定 |
| `gold.jsonl` | 参考标签，仅供本地评分 |
| `manifest.json` | 数据来源、校验和、样本数量及提示模板统计 |

唯一配置 [configs/deepseek.json](configs/deepseek.json) 使用 `data/criticlean-deepseek/author-prompts.jsonl`。从本项目根目录运行命令即可。

## 3. 配置密钥并检查环境

已经在当前终端导出过密钥时，无需重复设置。不要将密钥填进配置文件或提交到 GitHub。程序只读取 `DEEPSEEK_API_KEY` 环境变量，不自动加载 `.env`。

```zsh
read -s "DEEPSEEK_API_KEY?请粘贴 DeepSeek API Key，然后按回车："
export DEEPSEEK_API_KEY
```

```bash
lean-critic doctor --config configs/deepseek.json

lean-critic plan \
  --cases data/criticlean-deepseek/cases.jsonl \
  --config configs/deepseek.json
```

`doctor` 检查本地环境和密钥是否存在；`plan` 检查配置、输入及计划调用量。二者不调用模型，也不验证账户余额、密钥有效性或模型实际可用性。`run` 在创建输出目录前检查密钥是否缺失或含控制字符；这类本地问题立即退出。真实请求返回的身份验证错误仍记录为技术失败。

## 4. 先运行 5 条连接测试

```bash
lean-critic select \
  --prepared data/criticlean-deepseek \
  --out data/smoke5 \
  --limit 5

lean-critic run \
  --cases data/smoke5/cases.jsonl \
  --config configs/deepseek.json \
  --out runs/deepseek-smoke5-max-v2 &&

lean-critic evaluate \
  --run runs/deepseek-smoke5-max-v2 \
  --gold data/smoke5/gold.jsonl \
  --out reports/deepseek-smoke5-max-v2
```

`run` 开始真实付费调用，这个例子最多发出 5 次请求。此子集按准备数据顺序取前 5 条，仅用于检查连接、回复保存和标签提取，不是正式测试集，也没有训练／开发／测试划分。

先检查原始回复与技术失败数，确认流程正常，再运行全量。5 条上看到的比例不能代表 500 条总体表现。全量运行会再次调用这 5 条，两次实验的费用分别记录。

## 5. 运行正式 500 条实验

```bash
lean-critic run \
  --cases data/criticlean-deepseek/cases.jsonl \
  --config configs/deepseek.json \
  --out runs/deepseek-bench500-max-v2 &&

lean-critic evaluate \
  --run runs/deepseek-bench500-max-v2 \
  --gold data/criticlean-deepseek/gold.jsonl \
  --out reports/deepseek-bench500-max-v2
```

每题最多一次请求，全量最多 500 次。使用新的输出目录；不要把新结果写进以前自写 Prompt、程序过滤或多轮机制的运行目录。

0.2.1 默认保持 `high` 思考，将 `max_output_tokens` 提高到官方当前允许的 **393216 token（384K）**，是旧版 8192 的 48 倍。`request_timeout_seconds`、`max_wall_seconds_per_case` 与 `provider.max_response_bytes` 均设为 `null`，不再由客户端设置请求／每题时长上限或 2 MiB 响应大小上限；仍不自动重试。思考 token 包含在输出预算内。参数依据截至 2026-10-20 的 [DeepSeek 模型元数据文档](https://api-docs.deepseek.com/api/list-models/)。

这消除了旧版较小的本地限制，**不表示服务端无限输出，也不能保证所有请求一定给出标签**。服务端仍有 393216 输出 token 硬上限，也可能出现网络断开、服务错误、拒绝或未遵循格式。只省略 `max_output_tokens` 不代表无限，可能退回服务端默认值，所以本配置明确发送官方最大值。不附加 `temperature`、`top_p` 或 `seed`；接口不支持存储响应，程序不发送 `store` 参数。

程序记录接口可获得的 token 用量；无法获得的用量保留为未知。**没有美元费用硬上限，也没有预填单价；实际扣费以 DeepSeek 控制台为准。** 连接失败或本地超时不代表服务端一定没有处理或计费。

## 5a. 已跑完旧版 500 条：只重跑 52 条截断样本

旧版 `runs/deepseek-bench500-v1/` 与对应报告保持不变：8192 输出预算下，有 52 条在最终标签出现前达到输出上限，其中 49 条没有最终回复文本，3 条只有部分文本；**52 条都没有显式的 `Correct` / `Incorrect`**。这不是 DeepSeek 主动回答“无法判断”。旧版整体 ACC 为 389/500＝77.8%，不能被新参数结果覆盖。

对本工作区这次旧运行，已按原始 trace 的截断状态准备好 `data/deepseek-truncated52-v1/`，仅选择这 52 条，不依据公开标签选样。目录包含 cases、gold、原始消息和选择来源 manifest；干净 GitHub 源码不含这些本地实验文件。若没有这次旧运行和该目录，请使用上一节的完整 500 条命令，不自行假造该子集。

确认已安装 0.2.1 后，在项目根目录执行：

```bash
lean-critic plan \
  --cases data/deepseek-truncated52-v1/cases.jsonl \
  --config configs/deepseek.json &&

lean-critic run \
  --cases data/deepseek-truncated52-v1/cases.jsonl \
  --config configs/deepseek.json \
  --out runs/deepseek-truncated52-max-v2 &&

lean-critic evaluate \
  --run runs/deepseek-truncated52-max-v2 \
  --gold data/deepseek-truncated52-v1/gold.jsonl \
  --out reports/deepseek-truncated52-max-v2
```

唯一配置仍读取完整基准的 `data/criticlean-deepseek/author-prompts.jsonl`，按题目 ID 和哈希为所选 52 条匹配同一份原始消息；不需要修改配置或 Prompt。这段命令最多新增 52 次真实付费请求；`plan` 本身不调用模型。384K 是允许使用的最大量，并非每次都会生成到该长度，但最坏输出规模显著增大，单题也可能等待更久。

这份结果必须标为“旧运行截断的 52 条子集重跑”，其 ACC 不能称为 500 条成绩，也不能择优替换旧成绩。若要评测新预算下完整 500 条，单独使用上一节 `deepseek-bench500-max-v2` 的命令；这会发起另一批最多 500 次请求。

## 6. 查看 Prompt、回复和分数

先打开这些文件：

| 文件 | 查看内容 |
| --- | --- |
| `runs/deepseek-bench500-max-v2/author-messages.json` | 本次实际采用的逐题原始提示 |
| `runs/deepseek-bench500-max-v2/traces/*.json` | 单次请求、完整返回 JSON、最终回复文本、用量、错误和时间 |
| `runs/deepseek-bench500-max-v2/predictions.jsonl` | 每题提取的标签、理由与技术状态 |
| `runs/deepseek-bench500-max-v2/report.md` | 逐题结果概览 |
| `runs/deepseek-bench500-max-v2/config.json` | 本次配置快照，不含密钥值 |
| `runs/deepseek-bench500-max-v2/manifest.json` | 输入／配置／代码绑定、完成状态与累计用量的可读镜像 |
| `runs/deepseek-bench500-max-v2/checkpoint.json` | 权威原子检查点，和整个运行目录一并保留，勿手工修改 |
| `reports/deepseek-bench500-max-v2/summary.md` | 评分摘要 |
| `reports/deepseek-bench500-max-v2/per-case.csv` | 逐题公开标签与模型标签对照 |
| `reports/deepseek-bench500-max-v2/metrics.json` | 总体、作者标记编译通过子集及各原始标签的指标 |
| `reports/deepseek-bench500-max-v2/diagnostics.jsonl` | 评分时发现的缺失或不匹配记录 |

主指标为 **ACC = 与公开标签一致的题数 / 全部有标签题数**，读取 `metrics.json` 中的 `overall.classification_metrics.ACC.value`。

模型的 `Correct` 在内部记为 `supported`，`Incorrect` 记为 `contradicted`；这是字段映射，没有增加评审。API 失败或无法提取唯一标签记为技术失败（内部 `unknown`），保留在总分母中，不能当作模型主动弃权。

公开标签中，250 条为语义对应、200 条为编译通过但不对应、50 条为编译失败。除全部 500 条外，也报告作者标为编译通过的 450 条子集；这些编译标签没有在本项目重新验证。指标公式、混淆矩阵和失败解释见[结果说明](docs/results.md)。

## 7. 中断、失败和复现

同一次实验中断后，先使用相同命令并追加 `--resume`，再运行 `evaluate`。评分命令会拒绝尚未完成的运行或落后的结果镜像，避免静默评分不完整记录：

```bash
lean-critic run \
  --cases data/criticlean-deepseek/cases.jsonl \
  --config configs/deepseek.json \
  --out runs/deepseek-bench500-max-v2 \
  --resume
```

- 恢复要求输入、配置和实现等绑定一致；更新代码、改参数后使用新目录。
- 已完成的题目不会再次请求，包括已记录的技术失败。
- 已派发但未得到完整本地结果的请求可能已计费；恢复时记录为中断技术失败，不自动再次调用。
- 确需重跑失败题时，建立独立的新实验并清楚注明选择范围，不将多次结果择优合并成单次成绩。
- `status: "complete"` 表示遍历完成；`label_status: "complete"` 才表示每题都有有效模型标签。`valid_label_cases` 和 `technical_failure_cases` 给出两类数量。
- 如仍缺标签，`run` 保存全部记录后以退出码 **3** 结束，`&& evaluate` 会停止。需要统计包含技术失败的分数时，可以单独执行对应的 `evaluate` 命令；失败仍留在分母，不能因标签不全而隐藏。

`checkpoint.json` 是恢复的权威记录；`manifest.json` 和 `predictions.jsonl` 是便于阅读与评分的镜像。恢复时只修复能与已提交检查点绑定的中断写入，不接受任意手工修改。

准备和运行时会验证数据格式及哈希绑定。这些验证保证输入和记录可追溯，不判断数学理由是否正确。不要手工修改已完成运行内的 Prompt、预测或 traces；复现实验应保留完整目录和版本信息。密钥不写入快照，模型别名背后的服务端版本变化仍无法由本地哈希冻结。

## 工程结构

```text
lean-critic-lab/
├── configs/deepseek.json       # 唯一正式实验配置
├── src/lean_critic/            # 数据、单次调用、标签解析、评分
├── tests/                     # 离线测试
├── scripts/                   # 离线 smoke 与干净源码导出
├── docs/protocol.md            # 来源、保持项及复现边界
├── docs/results.md             # 标签、指标与失败解释
├── data/                      # 本地数据，不提交
├── runs/                      # 原始实验记录，可纳入版本控制
├── reports/                   # 评估结果，可纳入版本控制
├── RELEASE_FILES.txt          # 发布源码白名单
└── Makefile
```

早期自写提示、证据过滤与多阶段机制的记录属于历史条件，不能混入当前原始提示协议的成绩。原始提示协议的旧版 `deepseek-bench500-v1` 则是有效的 8192 预算实验，必须原样保留；0.2.1 的最大预算属于新的推理条件，使用独立 v2 目录。离线验证没有产生新的 DeepSeek 检测分数。本次重构前的源码与旧运行记录另存于当前工作区的同级目录 `../lean-critic-lab-archive/2026-10-19-before-deepseek-only/`；导出的 GitHub 源码不包含这份本地归档。

## 离线检查和 GitHub 提交

```bash
python -m pip install '.[dev]'
make check
make release
make github-dir
```

`make check` 不调用真实 API，验证数据处理、原提示保留、标签提取、失败处理及评分流程；通过不代表模型有良好检测效果。`make release` 按 `RELEASE_FILES.txt` 导出 `releases/lean-critic-lab.zip` 及校验信息；`make github-dir` 导出同一白名单的干净源码目录 `releases/lean-critic-lab-github/`，其中没有 `.git`。

当前项目已初始化为独立 Git 仓库并配置远程地址时，可以直接提交运行记录和评估结果，无需重复导出或初始化：

```bash
git add .gitignore README.md README.en.md NOTICE runs reports
git status --short
git commit -m "Add experiment records and evaluation results"
git push
```

**下面的导出流程适用于另建仅含干净源码的全新仓库。** 如果源码位于更大的父 Git 仓库里，先导出再从独立目录提交，避免带上父项目。下面的目标目录应尚不存在：

```bash
mkdir -p ~/Projects
cp -R releases/lean-critic-lab-github ~/Projects/lean-critic-lab
cd ~/Projects/lean-critic-lab

git init
git add .
git status --short
git commit -m "Add DeepSeek replacement evaluation for CriticLeanBench"
```

先在 GitHub 网页创建一个空仓库，不自动添加 README／许可证／`.gitignore`。将以下占位地址换成你的仓库地址：

```bash
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/lean-critic-lab.git
git push -u origin main
```

`.gitignore` 允许提交 `runs/` 和 `reports/`，仍排除环境目录（如 `.venv`）、`data/`、`releases/`、密钥文件及旧工程归档。`make release` 和 `make github-dir` 仍按 `RELEASE_FILES.txt` 白名单仅导出源码，不包含 `runs/` 或 `reports/`。提交前检查 `git status`，不要使用强制添加来绕过排除规则。发布结果时保留完整实验记录，并注明模型、日期、配置、样本数、技术失败数及数据版本；本工程不会自动上传任何内容。

原创代码和文档使用 MIT；第三方数据和引用内容保留各自许可，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

来源：[论文](https://arxiv.org/abs/2507.06181) · [作者仓库](https://github.com/multimodal-art-projection/CriticLean) · [固定版本数据](https://huggingface.co/datasets/m-a-p/CriticLeanBench/tree/6225522bce304f62d4907aedb9642255f9417bf0) · [DeepSeek Responses API](https://api-docs.deepseek.com/guides/responses_api/)
