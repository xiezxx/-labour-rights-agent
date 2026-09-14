"""
劳动维权咨询 Agent —— LangGraph 版本
========================================

与 labour_agent.py（手写循环版）相同的 4 个工具与业务逻辑，
编排层换成 LangGraph StateGraph：

             ┌──────────────────────────────────┐
             │  llm 节点：模型思考并决定下一步   │
             └──────┬────────────────┬──────────┘
             有 tool_calls       无 tool_calls
                    │                 │
                    ▼                 ▼
             ┌─────────────┐         END（最终回答）
             │ tools 节点  │
             │ 执行工具调用 │
             └──────┬──────┘
                    └────────→ 回到 llm 节点

工具实现直接 import 自 labour_agent.py——演示"编排层换框架，
业务工具不用动"的分层设计。

护栏：recursion_limit=19 限制图递归深度（约 8 轮模型调用），
超过后 LangGraph 抛 GraphRecursionError，循环被强制终止。

运行（独立虚拟环境，避免污染论文项目依赖）：
    python -m venv .venv
    .venv/Scripts/pip install openai python-dotenv langgraph langchain-openai
    .venv/Scripts/python langgraph_agent.py --demo
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Literal

try:
    sys.stdout.reconfigure(encoding="utf-8")
    # stdin 同样显式指定，避免管道/IDE 下中文输入按 GBK 解码产生 surrogate 字符
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env", override=False)
load_dotenv(override=False)

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 复用 v1 的纯工具实现与场景脚本
from labour_agent import (
    DEMO_SCENARIOS,
    SYSTEM_PROMPT,
    calculate_compensation as _calculate_compensation,
    check_arbitration_deadline as _check_deadline,
    search_law as _search_law,
)

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
RECURSION_LIMIT = 19  # 图递归护栏：llm+tools 一轮约 2 个 superstep，19 ≈ 最多 8 轮模型调用

# 脚本演示的自动应答队列（交互模式为 None）
SCRIPTED_ANSWERS = None

# 全局静默开关（agent_eval.py 评测时置 False，避免刷屏）
VERBOSE = True


# ════════════════════════════════════════════════════════════════
# 1. 工具（@tool 装饰器：docstring 就是给模型看的工具描述）
# ════════════════════════════════════════════════════════════════
@tool
def search_law(query: str) -> str:
    """检索劳动法律法规条文。回答中引用的法条必须来自本工具返回结果，严禁凭记忆编造条文号或条文内容。"""
    return _search_law(query)


@tool
def calculate_compensation(
    years: float,
    monthly_wage: float,
    reason: Literal["违法解除", "无过失性辞退", "无过失性辞退未提前30日通知", "过失性辞退", "协商解除（单位提出）", "其他"],
) -> str:
    """按《劳动合同法》第47/87/40条计算经济补偿（N）、代通知金情形（N+1）或违法解除赔偿金（2N）。涉及任何金额都必须调用本工具，严禁自行心算。"""
    return json.dumps(_calculate_compensation(years, monthly_wage, reason), ensure_ascii=False)


@tool
def check_arbitration_deadline(dispute_type: Literal["劳动报酬", "其他"]) -> str:
    """查询劳动仲裁申请时效规则（一年时效及其例外）。用户询问'还能不能要回来''是否过期'等问题时调用。"""
    return _check_deadline(dispute_type)


@tool
def ask_user(question: str) -> str:
    """案情关键信息不足时向用户追问（如工龄、月工资、辞退理由、是否签合同）。一次最多问两个问题，不要猜测关键数字。"""
    if VERBOSE:
        print(f"    ❓ {question}")
    if SCRIPTED_ANSWERS is not None:
        if SCRIPTED_ANSWERS:
            answer = SCRIPTED_ANSWERS.pop(0)
            if VERBOSE:
                print(f"    🧑（脚本自动回答）{answer}")
            return answer
        return "（脚本未提供更多回答，请基于已有信息给出结论或建议）"
    try:
        answer = input("    🧑 你的回答：").strip()
    except EOFError:
        return "（无法获取用户输入，请基于已有信息给出结论或建议）"
    return answer or "（用户未回答，请基于已有信息给出结论或建议）"


TOOLS = [search_law, calculate_compensation, check_arbitration_deadline, ask_user]

llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0.2)
llm_with_tools = llm.bind_tools(TOOLS)


# ════════════════════════════════════════════════════════════════
# 2. 状态图：状态 = 消息列表，节点 = llm / tools
# ════════════════════════════════════════════════════════════════
class AgentState(dict):
    messages: Annotated[list, add_messages]


def call_llm(state: AgentState) -> dict:
    """llm 节点：模型自主决定——继续调用工具，还是给出最终回答"""
    if VERBOSE:
        print("  ⏳ llm 节点：模型思考中…")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_continue(state: AgentState) -> str:
    """条件边（路由逻辑）：最后一条消息带 tool_calls → 去 tools 节点；否则 → END"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


builder = StateGraph(AgentState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(TOOLS))  # 预置节点：自动执行 AIMessage 里的工具调用
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")
graph = builder.compile()


# ════════════════════════════════════════════════════════════════
# 3. 运行：stream 逐节点输出，运行时能看清图上的每一步
# ════════════════════════════════════════════════════════════════
def run_turn(messages: list, question: str, verbose: bool = True) -> tuple:
    """执行一轮 Agent 对话。

    Returns:
        (messages, stats) —— stats 含 answer / tool_calls / tool_outputs / latency / recursion_hit
        verbose=False 时静默执行，供 agent_eval.py 批量评测使用
    """
    import time

    if verbose:
        print(f"\n👤 用户：{question}")
    messages.append(HumanMessage(content=question))
    config = {"recursion_limit": RECURSION_LIMIT}
    final_answer = None
    tool_names = []
    tool_outputs = []
    recursion_hit = False
    t0 = time.time()

    try:
        for chunk in graph.stream({"messages": list(messages)}, config=config, stream_mode="updates"):
            for _node_name, update in chunk.items():
                for msg in update.get("messages", []):
                    messages.append(msg)
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            tool_names.append(tc["name"])
                            if verbose:
                                print(f"  🔧 调用工具：{tc['name']}({json.dumps(tc.get('args', {}), ensure_ascii=False)})")
                    elif isinstance(msg, ToolMessage):
                        content = msg.content or ""
                        tool_outputs.append(content)
                        if verbose:
                            preview = content[:150].replace("\n", " ")
                            print(f"  📥 工具结果：{preview}{'…' if len(content) > 150 else ''}")
                    elif isinstance(msg, AIMessage):
                        final_answer = msg.content or ""
    except GraphRecursionError:
        recursion_hit = True
        if verbose:
            print("\n⚠️ 达到图递归上限（recursion_limit=19），循环被 LangGraph 强制终止——防失控护栏。")

    latency = time.time() - t0
    answer = final_answer or "（未生成最终回答）"
    if verbose:
        print("\n🤖 回答：\n" + answer)
        print("\n" + "─" * 50)
        print(f"📊 本次运行：{len(tool_names)} 次工具调用"
              + (f"，序列：{', '.join(tool_names)}" if tool_names else "")
              + f"，耗时 {latency:.1f}s")
        print("   （工具调用决策由模型在 llm 节点自主做出，图的边只负责路由，不规定调用什么）")
        print("─" * 50)

    return messages, {
        "answer": answer,
        "tool_calls": tool_names,
        "tool_outputs": tool_outputs,
        "latency": latency,
        "recursion_hit": recursion_hit,
    }


def main():
    print("=" * 60)
    print("  劳动维权咨询 Agent —— LangGraph 版本")
    print("  StateGraph 编排（llm ⇄ tools 循环）+ 4 工具")
    print("=" * 60)

    if not API_KEY:
        print("\n❌ 未找到 OPENAI_API_KEY。请在 .env 中配置（见文件头部说明）。")
        return

    global SCRIPTED_ANSWERS

    if "--demo" in sys.argv:
        for i, scenario in enumerate(DEMO_SCENARIOS, 1):
            print(f"\n\n{'=' * 60}\n🎬 {scenario['title']}\n{'=' * 60}")
            SCRIPTED_ANSWERS = list(scenario["answers"])
            run_turn([SystemMessage(content=SYSTEM_PROMPT)], scenario["question"])
        print("\n🎬 演示结束。交互模式请运行：.venv/Scripts/python langgraph_agent.py")
        return

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    print("\n直接输入你的劳动法问题，可以多轮追问，直接回车退出。\n")
    while True:
        try:
            question = input("👤 你：").strip()
        except EOFError:
            break
        if not question:
            break
        messages, _stats = run_turn(messages, question)
    print("👋 再见")


if __name__ == "__main__":
    main()
