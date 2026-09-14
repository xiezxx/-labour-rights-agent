# 本地测试指南

给作者自己的操作手册：**怎么跑、看什么、出问题怎么办**。
（Windows 命令用 Git Bash 写法；PowerShell / cmd 把 `/` 换成 `\` 即可，如 `.venv\Scripts\python.exe`）

---

## 第 0 步：环境自检（30 秒，先跑这个）

```bash
cd "D:/My wordl four/labour-agent-demo"
.venv/Scripts/python selftest.py
```

会逐项检查 Python、依赖、`.env`、状态文件、模块导入、API 连通性，最后给出结论。

- 全部 ✅ → 直接进第 1 步
- 有 ⚠️/❌ → 按提示处理（缺依赖就 `pip install`，Key 没填就编辑 `.env`）

想省掉那次 API 调用（约 ¥0.001）：`selftest.py --no-api`

---

## 第 1 步：看演示怎么跑（v3 完整功能，约 1 分钟）

```bash
.venv/Scripts/python agent_hitl.py --thread demo-mine --demo --trace
```

**这一条命令能同时看到四个能力**，按顺序观察：

| 阶段 | 你会看到 | 说明 |
|---|---|---|
| 1 | `🆕 新线程「demo-mine」` | 记忆库初始化 |
| 2 | `❓` 追问 + `🧑（脚本自动回答）` | 反问澄清（脚本模拟你回答） |
| 3 | `🛑 人在环审批` 弹出文书要素 | **审批闸门**——生成正式文书前暂停 |
| 4 | `📥 工具结果：劳动人事争议仲裁申请书…` | 批准后生成文书 |
| 5 | `🧠 从记忆恢复：已有 N 条历史消息` | **记忆生效**——第二轮没重述案情 |
| 6 | `📈 可观测性报告` | token 用量、缓存命中、耗时、成本 |

**想试"否决"路径**（看 Agent 按你的意见重做）：

```bash
.venv/Scripts/python agent_hitl.py --thread demo-reject --demo --approve no --trace
```

Agent 会收到"赔偿金额计算依据再补充一下"这条意见，重新检索法条补强依据，然后**再次触发审批**（脚本会作出第二次决定：批准，于是生成补上计算依据的修订版文书）。整条链路是：否决 → 补工具回执 → 带意见重做 → 二次批准。

---

## 第 2 步：验证跨进程记忆（最有说服力的一步）

**关键**：两次运行是**两个独立进程**，模拟"关掉程序明天再打开"。

```bash
# 第一次：报案情（上面已经跑过 demo-mine 了，这里换个线程名从零开始）
.venv/Scripts/python agent_hitl.py --thread case-zhang --demo

# 第二次：全新进程，不重述案情，直接问
.venv/Scripts/python agent_hitl.py --thread case-zhang --question "我的工龄是几年？月工资多少？被申请人是谁？"
```

**预期**：第二次开场打印 `🧠 从记忆恢复：线程「case-zhang」已有 N 条历史消息`，
然后准确答出 张三 / 某某科技有限公司 / 2年半 / 9,000 元 —— **你一个字都没重述**。

查看某个线程到底存了什么：

```bash
.venv/Scripts/python agent_hitl.py --thread case-zhang --show-memory
```

---

## 第 3 步：交互模式（自己当用户，最真实）

```bash
.venv/Scripts/python agent_hitl.py --thread my-test --trace
```

然后在 `👤 你：` 后面输入。**推荐这几个问题，覆盖不同工具路径**：

| 输入 | 观察重点 |
|---|---|
| `公司没有任何理由把我辞退了，我能拿多少赔偿？` | 会反问你姓名/工龄/工资——**故意只回答一半**，看它继续追问而不是瞎猜 |
| `公司拖欠我3年工资，我还能要回来吗？` | 调用仲裁时效工具，命中第 27 条"存续期间不受一年限制"的例外 |
| `我周末经常加班，公司不给加班费，合法吗？` | 检索到 150%/200%/300% 的标准 |
| `我在工作中受伤了，算工伤吗？` | **知识边界测试**——迷你法条库没有工伤保险条例，看它会不会编造条文 |
| `帮我写一份仲裁申请书` | 触发**审批闸门**（前提：先告诉它你的姓名、公司、工龄、工资） |

**审批时怎么答**（出现 `🛑 人在环审批` 之后）：

| 你输入 | 效果 |
|---|---|
| `y` | 批准 → 生成正式文书 |
| `n` | 否决（不给意见）→ Agent 会问你想改什么 |
| 直接打一段文字 | 否决 + 把这段文字当修改意见 → Agent 据此重做 |

其他命令：`/memory` 查看本线程记忆 · 直接回车退出

---

## 第 4 步：跑评测（串行执行约 16 分钟，会花几毛钱）

```bash
.venv/Scripts/python agent_eval.py                 # 全量 11 案 × 2 架构
.venv/Scripts/python agent_eval.py --case C01 --merge  # 只跑一个用例（约 1 分钟）并合并进已有结果
.venv/Scripts/python agent_eval.py --rescore        # 只重算指标，不调 LLM（不花钱）
```

预期：C01 用例 Agent 版应拿到"工具召回 100% / 反问 ✅ / 金额 ✅ / 引用可验证 100%"。
明细写入 `eval_results.json`（含双方完整答案，可逐条复核）。
**注意**：不加 `--merge` 的 `--case` 运行会覆盖该文件，导致其余用例的明细丢失。

> 改过打分逻辑后，**先用 `--rescore` 离线重算**验证口径，别急着重跑全量。

---

## 第 5 步：其他版本（对比用）

```bash
.venv/Scripts/python labour_agent.py --demo          # v1 手写循环（零框架）
.venv/Scripts/python langgraph_agent.py --demo       # v2 LangGraph（可见并行工具调用）
.venv/Scripts/python observability.py                # 可观测性自检（不调 LLM，验证成本算法）
```

---

## 快速演示脚本（3 分钟版）

时间紧就用这两条，一次看全所有能力：

```bash
# ① 完整能力一览（记忆 + 审批 + 可观测性），约 1 分钟
.venv/Scripts/python agent_hitl.py --thread demo --demo --trace

# ② 跨进程记忆（全新进程，不重述案情）
.venv/Scripts/python agent_hitl.py --thread demo --question "被申请人是谁？"
```

重点在于展示**为什么这么设计**，而不是念输出（各版本的设计取舍见 README 对应章节）。

---

## 常见问题

**Q：为什么要用 `.venv/Scripts/python` 而不是直接 `python`？**
A：`.venv` 是项目专属虚拟环境，装了 langgraph 等依赖。直接用系统 `python` 会缺包，或污染其他项目。
自检脚本会提示你当前用的是哪个解释器。

**Q：测试需要开代理（VPN）吗？**
A：**不需要**。调用的是 DeepSeek（国内可直连）。只有两种情况要代理：① 推送 GitHub（走 `-c http.proxy=http://127.0.0.1:7893`）② 开启 LangSmith 云端追踪。

**Q：跑一次花多少钱？**
A：单轮问答约 ¥0.005–0.01（输入大部分命中缓存，命中价只有未命中的 1/120）。跑完整评测约几毛钱。加 `--trace` 能看到每次的确切估算值。

**Q：终端里中文/emoji 变成乱码怎么办？**
A：各脚本顶部都有 `sys.stdout.reconfigure(encoding="utf-8")`，正常情况下不会乱码。若仍乱码，在 Git Bash 里执行 `chcp.com 65001`，或用 Windows Terminal 替代 cmd。

**Q：想清空记忆从头开始？**
A：删掉 `agent_memory.sqlite` 即全部清空；只想清一个会话就换 `--thread` 名字。清追踪日志删 `trace_log.jsonl`。
（两个文件连同 SQLite 的 `-wal`/`-shm` 边车文件都在 `.gitignore` 里，不会进仓库。）

**Q：模型名怎么改？**
A：编辑 `.env` 的 `OPENAI_MODEL`。换模型后记得核对 `observability.py` 里 `PRICE_TABLE` 的单价，或用 `LLM_PRICE_*` 环境变量覆盖。

**Q：报错 `OPENAI_API_KEY 未填写`？**
A：`.env` 没配好。复制 `.env.example` 为 `.env`，填入你的 Key。
