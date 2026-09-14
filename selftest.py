"""
环境自检 —— 一条命令确认项目是否就绪
========================================

检查项：
  1. Python 版本与依赖包（openai / langgraph / langchain-openai / sqlite 检查点 / langsmith）
  2. .env 配置（API Key、模型名、Base URL）
  3. 本地状态文件（记忆库 agent_memory.sqlite、追踪日志 trace_log.jsonl）
  4. API 连通性（默认发一次最小请求实测；加 --no-api 跳过，不花钱）
  5. 各模块能否正常导入

用法：
    .venv/Scripts/python selftest.py            # 完整自检（含一次最小 API 调用）
    .venv/Scripts/python selftest.py --no-api    # 跳过 API 调用
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent

OK, WARN, FAIL = "✅", "⚠️ ", "❌"
problems = []


def check_python():
    v = sys.version_info
    print(f"{OK if v >= (3, 10) else FAIL} Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        problems.append("Python 版本过低，建议 3.10+")
    print(f"  解释器：{sys.executable}")
    if ".venv" not in sys.executable:
        print(f"{WARN}当前用的不是项目虚拟环境，建议用 .venv/Scripts/python 运行")


def check_deps():
    import importlib.metadata as meta
    required = {
        "openai": "调用 LLM",
        "python-dotenv": "读取 .env",
        "langgraph": "图编排",
        "langchain-openai": "LLM 封装",
        "langchain-core": "回调机制",
        "langgraph-checkpoint-sqlite": "跨进程记忆",
        "langsmith": "可选：云端追踪（本地追踪不依赖它）",
    }
    for pkg, why in required.items():
        try:
            print(f"{OK} {pkg} {meta.version(pkg)}   （{why}）")
        except meta.PackageNotFoundError:
            print(f"{FAIL} {pkg} 未安装   （{why}）")
            problems.append(f"缺少依赖 {pkg}，执行：pip install {pkg}")


def check_env():
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR / ".env", override=False)
    import os

    env_file = SCRIPT_DIR / ".env"
    if not env_file.exists():
        print(f"{FAIL} 未找到 .env（请复制 .env.example 为 .env 并填入 Key）")
        problems.append("缺少 .env")
        return None, None, None

    print(f"{OK} .env 已存在（{env_file.stat().st_size} 字节）")

    key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "deepseek-v4-pro")
    base = os.getenv("OPENAI_BASE_URL", "")
    if not key or key.startswith("sk-你的"):
        print(f"{FAIL} OPENAI_API_KEY 未填写")
        problems.append("OPENAI_API_KEY 未填写")
    else:
        # 只显示前后各 4 位，避免完整密钥出现在终端输出里
        print(f"{OK} OPENAI_API_KEY 已配置（{key[:4]}…{key[-4:]}，长度 {len(key)}）")
    print(f"{OK if base else WARN} OPENAI_BASE_URL = {base or '（未设置，将用默认）'}")
    print(f"{OK} OPENAI_MODEL = {model}")

    if os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1"):
        print(f"{WARN} LangSmith 云端追踪已开启（数据会上传到境外，需配 HTTPS_PROXY）")
    else:
        print(f"{OK} LangSmith 未开启（本地 --trace 追踪不受影响）")
    return key, model, base


def check_state_files():
    memory = SCRIPT_DIR / "agent_memory.sqlite"
    if memory.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(str(memory))
            threads = conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()[0]
            conn.close()
            print(f"{OK} agent_memory.sqlite 已存在，含 {threads} 个会话线程（记忆可跨进程恢复）")
            print(f"   查看某个线程：python agent_hitl.py --thread <名字> --show-memory")
        except Exception as e:
            print(f"{WARN}agent_memory.sqlite 存在但读取失败：{str(e)[:80]}")
    else:
        print(f"{OK} agent_memory.sqlite 尚未创建（首次运行 agent_hitl.py 时自动生成）")

    trace = SCRIPT_DIR / "trace_log.jsonl"
    if trace.exists():
        lines = sum(1 for _ in open(trace, encoding="utf-8"))
        print(f"{OK} trace_log.jsonl 已存在，{lines} 条追踪记录")
    else:
        print(f"{OK} trace_log.jsonl 尚未创建（加 --trace 运行时生成）")


def check_imports():
    try:
        import labour_agent, langgraph_agent, agent_hitl, observability, agent_eval  # noqa
        print(f"{OK} 5 个模块导入正常（v1/v2/v3/可观测性/评测）")
    except Exception as e:
        print(f"{FAIL} 模块导入失败：{str(e)[:150]}")
        problems.append("模块导入失败")


def check_api(key, model, base):
    if not key:
        print(f"{FAIL} 跳过 API 测试（Key 未配置）")
        return
    print("   （发起一次最小请求实测，成本约 ¥0.001 以内…）")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base or None)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "请只回复两个字：正常"}], max_tokens=32,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        shown = repr(text) if text else "（无文本返回，但连接正常）"
        print(f"{OK} API 连通，模型 {model} 返回：{shown}"
              + (f"（本次 token：{usage.prompt_tokens}+{usage.completion_tokens}）" if usage else ""))
    except Exception as e:
        print(f"{FAIL} API 调用失败：{str(e)[:200]}")
        problems.append("API 调用失败（检查 Key、模型名、网络）")


def main():
    print("=" * 62)
    print("  劳动维权 Agent 演示项目 —— 环境自检")
    print("=" * 62)

    print("\n【1】Python 环境")
    check_python()

    print("\n【2】依赖包")
    check_deps()

    print("\n【3】配置（.env）")
    key, model, base = check_env()

    print("\n【4】状态文件")
    check_state_files()

    print("\n【5】模块导入")
    check_imports()

    print("\n【6】API 连通性")
    if "--no-api" in sys.argv:
        print(f"{OK} 已按 --no-api 跳过")
    else:
        check_api(key, model, base)

    print("\n" + "=" * 62)
    if problems:
        print(f"⚠️  发现 {len(problems)} 个问题，按顺序处理：")
        for i, p in enumerate(problems, 1):
            print(f"   {i}. {p}")
    else:
        print("🎉 全部就绪，可以开始测试。推荐第一步：")
        print("   .venv/Scripts/python agent_hitl.py --thread demo-mine --demo --trace")
    print("=" * 62)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
