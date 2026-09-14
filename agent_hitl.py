"""
劳动维权咨询 Agent —— v3：多轮记忆 + 人在环审批
=====================================================

在 v2（LangGraph 编排）基础上加两个生产级能力，都是落地生产环境时的关键能力：

1. **多轮记忆（持久化状态）**
   用 SqliteSaver 检查点把整个会话状态落盘，按 thread_id 隔离。
   效果：**关掉程序、重新打开，Agent 依然记得你的案情**（跨进程记忆），
   而不是只能在一轮对话内记住上下文。
   实现要点：graph.compile(checkpointer=SqliteSaver(conn))，
   每次调用传 config={"configurable": {"thread_id": ...}}。

2. **人在环审批（Human-in-the-loop）**
   生成《劳动仲裁申请书》这类**正式法律文书**属于高风险动作（用户可能拿去提交仲裁），
   不能让模型自主执行。流程：
       llm 决定调用 draft_arbitration_application
         → 【暂停】review_draft 节点调用 interrupt() 挂起图，把文书要素交给人类
         → 人类批准 → 执行工具生成文书 / 人类否决 → 带修改意见回到 llm 重做
   实现要点：LangGraph 的 interrupt() + Command(resume=...)，图必须有 checkpointer 才能挂起/恢复。

编排图（v2 基础上多一条审批分支）：

         ┌──────────────────────────┐
         │ llm 节点：模型思考并决策  │
         └──┬────────┬──────────┬───┘
      要写文书   要调其他工具   无工具调用
            │        │            │
            ▼        │            ▼
    ┌──────────────┐ │          END
    │ review 节点   │ │
    │ interrupt()   │ │
    │ 等待人工批准  │ │
    └───┬──────┬───┘ │
    批准 │      │ 否决 │
        ▼      └──────┴──→ 回 llm 节点（带人类修改意见）
    ┌───────────┐
    │ tools 节点 │
    └───────────┘

运行：
    # 交互模式（同一个 --thread 即同一份记忆）
    .venv/Scripts/python agent_hitl.py --thread zhang-san

    # 跨进程记忆演示：跑两次，第二次换问题，看它是否记得案情
    .venv/Scripts/python agent_hitl.py --thread demo-01 --demo     # 第一次：报案情 + 生成文书（需批准）
    .venv/Scripts/python agent_hitl.py --thread demo-01 --question "那仲裁时效是多久？"

    # 脚本化单问
    .venv/Scripts/python agent_hitl.py --question "公司拖欠我3年工资还能要回来吗？"
"""

import argparse
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Annotated, Literal

try:
    sys.stdout.reconfigure(encoding="utf-8")
    # stdin 也要显式指定：管道/IDE 运行窗口下若按 Windows 默认 GBK 解码，
    # 中文输入会产生 lone surrogate 字符，后续打印直接抛 UnicodeEncodeError
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
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from labour_agent import (
    SYSTEM_PROMPT,
    calculate_compensation as _calculate_compensation,
    check_arbitration_deadline as _check_deadline,
    search_law as _search_law,
)
from observability import TraceCollector

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
RECURSION_LIMIT = 25
MEMORY_DB = SCRIPT_DIR / "agent_memory.sqlite"  # 记忆落盘位置

VERBOSE = True
SCRIPTED_ANSWERS = None  # ask_user 的脚本应答队列


# ════════════════════════════════════════════════════════════════
# 1. 工具（前 4 个与 v2 相同，第 5 个是需审批的敏感操作）
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


# ToolNode 会用线程池并发执行同一批 tool_calls，而 ask_user 要读写"问题—回答"队列：
# 加锁串行化，保证每个问题拿到的回答不会错配到另一个问题
_ASK_LOCK = threading.Lock()


@tool
def ask_user(question: str) -> str:
    """案情关键信息不足时向用户追问（如工龄、月工资、辞退理由、是否签合同）。一次最多问两个问题，不要猜测关键数字。"""
    with _ASK_LOCK:
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


@tool
def draft_arbitration_application(
    applicant_name: str,
    respondent_name: str,
    claims_summary: str,
    amount: float,
    facts_summary: str,
) -> str:
    """生成《劳动人事争议仲裁申请书》正式文书。

    【敏感操作】本操作会产出用户可能直接提交仲裁委员会的正式法律文书，因此系统会先暂停、
    由用户人工核对要素并批准后才真正生成。用户明确要求"写申请书/申请劳动仲裁"时才调用；
    调用前须已确认申请人姓名、被申请人（公司）名称、请求事项、金额、事实经过。
    """
    # 确定性模板：正式文书的骨架由 Python 拼装，不让模型自由发挥格式
    today = time.strftime("%Y年%m月%d日")
    return f"""劳动人事争议仲裁申请书

申请人：{applicant_name}
被申请人：{respondent_name}

一、仲裁请求
{claims_summary}
请求金额合计：人民币 {amount:,.2f} 元

二、事实与理由
{facts_summary}

三、证据清单
1. 劳动合同或工作证明（证明劳动关系及起止时间）
2. 银行工资流水（解除或终止前十二个月，用于计算补偿基数）
3. 解除/终止劳动合同通知书或相关沟通记录（微信、邮件、录音）
4. 考勤记录、排班表（如有加班费争议）
5. 其他与本案有关的证据

此致
＿＿＿＿劳动人事争议仲裁委员会

申请人（签名）：{applicant_name}
{today}

（本文书由系统按案情要素生成，请在提交前自行核对全部内容，必要时咨询执业律师）"""


TOOLS = [search_law, calculate_compensation, check_arbitration_deadline, ask_user,
         draft_arbitration_application]
SENSITIVE_TOOL = "draft_arbitration_application"  # 需要人工审批的工具

# LLM 客户端懒加载：import 时不构造。否则"没有 .env 的全新 clone"连模块都导不进来
# （ChatOpenAI 缺 API Key 时会立即抛 OpenAIError），一行报错就劝退了看代码的人
_LLM_WITH_TOOLS = None


def get_llm():
    """按需构造 LLM（首次调用时初始化，之后复用）"""
    global _LLM_WITH_TOOLS
    if _LLM_WITH_TOOLS is None:
        _LLM_WITH_TOOLS = ChatOpenAI(
            model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0.2
        ).bind_tools(TOOLS)
    return _LLM_WITH_TOOLS


# ════════════════════════════════════════════════════════════════
# 2. 状态与节点
# ════════════════════════════════════════════════════════════════
class AgentState(dict):
    messages: Annotated[list, add_messages]   # 由 add_messages 归约器管理（含历史记忆）
    pending_draft: dict                       # 待审批的文书要素（含 tool_call_id，便于否决时补回执）
    review: dict                              # 审批结果 {approved, feedback}


def repair_dangling_tool_calls(messages: list) -> list:
    """返回补好回执的完整消息列表：为"调用了却没有结果"的 tool_call 生成占位 ToolMessage。

    递归上限强制中断、或审批中断未决时进程被关掉，都会在检查点里留下这种悬空调用。
    不补的话，下一轮把历史发给 LLM 会被 API 以 400 拒绝（assistant 的 tool_calls 必须
    紧跟对应的 tool 消息），而且该线程之后每次提问都会崩、永久不可用。

    两个关键点（都是踩过的坑）：
    1. 占位回执必须**插在该 tool_calls 消息之后**，不能追加到列表末尾，否则仍然 400；
    2. 只在构造请求时使用、**不写回状态**——写回会被 add_messages 归约器追加到末尾，
       位置反而是错的，还会让后续轮次误以为"已有回执"而不再修补。
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    repaired = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        repaired.append(m)
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            # 先原样保留紧随其后的已有回执（不打乱顺序），再在同一位置补齐缺失的回执
            j = i + 1
            while j < n and isinstance(messages[j], ToolMessage):
                repaired.append(messages[j])
                j += 1
            for tc in m.tool_calls:
                if tc["id"] not in answered:
                    repaired.append(ToolMessage(
                        content="（该工具调用未执行完成——可能因循环步数上限或人工审批中断，未产生结果）",
                        tool_call_id=tc["id"],
                    ))
                    answered.add(tc["id"])
            i = j
            continue
        i += 1
    return repaired


def call_llm(state: AgentState) -> dict:
    """llm 节点：模型自主决策。若决定生成正式文书，把要素暂存到 pending_draft 交给审批节点。"""
    if VERBOSE:
        print("  ⏳ llm 节点：模型思考中…")
    repaired = repair_dangling_tool_calls(state["messages"])
    added = len(repaired) - len(state["messages"])
    if added and VERBOSE:
        print(f"  🩹 检测到 {added} 个无回执的工具调用，已在本次请求中补占位回执")
    msg = get_llm().invoke(repaired)

    pending = {}
    for tc in getattr(msg, "tool_calls", None) or []:
        if tc["name"] == SENSITIVE_TOOL:
            pending = {
                "args": tc.get("args", {}),
                # 整批 tool_calls 的 id：否决时要为每个 id 补一条 ToolMessage，
                # 否则历史里存在"无回执的 tool_call"，再次调用 LLM 会被 API 拒绝
                "tool_call_ids": [c["id"] for c in msg.tool_calls],
            }
            break
    return {"messages": [msg], "pending_draft": pending}


def review_draft(state: AgentState) -> dict:
    """审批节点：调用 interrupt() 挂起整张图，把决定权交给人类。

    interrupt() 首次执行会中断图并保存检查点；人类给出决定后，
    用 Command(resume=决定) 恢复，本节点从头重跑，interrupt() 返回人类的决定。
    """
    pending = state.get("pending_draft") or {}
    decision = interrupt({
        "action": SENSITIVE_TOOL,
        "message": "Agent 准备生成正式《劳动仲裁申请书》，请核对要素后批准",
        "draft": pending.get("args", {}),
    })

    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        feedback = decision.get("feedback") or ""
    else:
        approved, feedback = bool(decision), ""

    out = {"review": {"approved": approved, "feedback": feedback}, "pending_draft": {}}

    if not approved:
        # 否决：为整批工具调用补回执（保持历史合法），并把人类意见带回给模型
        msgs = [
            ToolMessage(content="操作被用户否决，未执行。", tool_call_id=tcid)
            for tcid in pending.get("tool_call_ids", [])
        ]
        msgs.append(HumanMessage(
            content=f"【用户否决了文书生成】修改意见：{feedback or '（未说明，请询问用户想改什么）'}"
        ))
        out["messages"] = msgs
    return out


def after_llm(state: AgentState) -> str:
    """路由：要写文书 → 先审批；其他工具 → 直接执行；无工具调用 → 结束"""
    if state.get("pending_draft"):
        return "review"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def after_review(state: AgentState) -> str:
    """路由：批准 → 执行工具；否决 → 回 llm 按人类意见重做"""
    return "tools" if (state.get("review") or {}).get("approved") else "llm"


builder = StateGraph(AgentState)
builder.add_node("llm", call_llm)
builder.add_node("review", review_draft)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", after_llm, {"review": "review", "tools": "tools", END: END})
builder.add_conditional_edges("review", after_review, {"tools": "tools", "llm": "llm"})
builder.add_edge("tools", "llm")

# 关键：带上检查点存储器，图才能在 interrupt 处挂起、并在新进程里恢复记忆
_conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(_conn))


# ════════════════════════════════════════════════════════════════
# 3. 运行：流式打印 + 中断循环
# ════════════════════════════════════════════════════════════════
def _stream(inp, config: dict) -> bool:
    """跑一段图，逐节点打印产出（含工具调用与结果）。

    Returns:
        True 表示因触及递归上限被强制中断（调用方必须停止后续中断循环）
    """
    try:
        for chunk in graph.stream(inp, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__" or not isinstance(update, dict):
                    continue
                for msg in update.get("messages") or []:
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            print(f"  🔧 调用工具：{tc['name']}({json.dumps(tc.get('args', {}), ensure_ascii=False)[:200]})")
                    elif isinstance(msg, ToolMessage):
                        content = msg.content or ""
                        preview = content[:180].replace("\n", " ")
                        print(f"  📥 工具结果：{preview}{'…' if len(content) > 180 else ''}")
    except GraphRecursionError:
        print("\n⚠️ 达到图递归上限，循环被强制终止——防失控护栏。")
        return True
    return False


def _prompt_approval(payload: dict, scripted=None) -> dict:
    """人在环：把待批准要素展示给人类，收集决定"""
    print("\n" + "!" * 62)
    print("🛑 人在环审批：Agent 请求生成正式法律文书（高风险动作，已暂停）")
    print("!" * 62)
    for k, v in (payload.get("draft") or {}).items():
        print(f"   {k}：{v}")
    print("!" * 62)

    if scripted is not None:
        # 脚本队列用尽时**默认否决**：高风险动作的审批绝不能因为"没人回答"就放行
        decision = scripted.pop(0) if scripted else {
            "approved": False,
            "feedback": "（脚本审批队列已用尽，按安全默认值否决；需批准请补充审批决定）",
        }
        tag = "批准" if decision.get("approved") else "否决"
        print(f"   （脚本自动决策：{tag}"
              + (f"，意见：{decision.get('feedback')}" if decision.get("feedback") else "") + "）")
        return decision

    try:
        ans = input("   批准生成？(y=批准 / n=否决 / 直接输入文字=否决并附修改意见)：").strip()
    except EOFError:
        return {"approved": False, "feedback": "无法获取用户输入"}
    if ans.lower() in ("y", "yes", "是", "批准", "1"):
        return {"approved": True}
    if ans.lower() in ("n", "no", "否", "否决", "0"):
        return {"approved": False, "feedback": ""}
    return {"approved": False, "feedback": ans}


def run_turn(thread_id: str, question: str, scripted_approvals=None, verbose: bool = True,
             callbacks: list = None) -> dict:
    """执行一轮对话（同一 thread_id = 同一份持久记忆）"""
    global VERBOSE
    VERBOSE = verbose
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT,
              "callbacks": callbacks or []}  # 可观测性回调在此注入，与检查点互不影响

    # —— 记忆恢复：检查点里已有历史就不必再塞 system prompt ——
    snapshot = graph.get_state(config)
    prior = (snapshot.values or {}).get("messages") or []
    if prior:
        print(f"\n🧠 从记忆恢复：线程「{thread_id}」已有 {len(prior)} 条历史消息（来自 {MEMORY_DB.name}）")
        inputs = {"messages": [HumanMessage(content=question)]}
    else:
        print(f"\n🆕 新线程「{thread_id}」，本次对话将写入 {MEMORY_DB.name}")
        inputs = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]}

    if verbose:
        print(f"\n👤 用户：{question}")
    t0 = time.time()
    hit_limit = _stream(inputs, config)

    # —— 中断循环：图若停在审批点，就收集人类决定后恢复 ——
    approvals = 0
    while not hit_limit:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        tasks = getattr(snapshot, "tasks", ()) or ()
        interrupts = getattr(tasks[0], "interrupts", ()) if tasks else ()
        if not interrupts:
            # 有待办任务但没有真实审批请求（递归上限留下的 pending task 就是这种）：
            # 没有任何人在等回答，必须跳出，否则会对着不存在的审批无限弹窗/忙等
            print("\n⚠️ 检查点存在未完成任务但无审批请求，本轮结束（可用同一 --thread 继续）。")
            break
        decision = _prompt_approval(interrupts[0].value or {}, scripted_approvals)
        approvals += 1
        hit_limit = _stream(Command(resume=decision), config)

    # —— 取最终回答 ——
    final = ""
    for msg in reversed((graph.get_state(config).values or {}).get("messages") or []):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and msg.content:
            final = msg.content
            break

    if verbose:
        print("\n🤖 回答：\n" + (final or "（未生成最终回答）"))
        total = len((graph.get_state(config).values or {}).get("messages") or [])
        print("\n" + "─" * 62)
        print(f"📊 线程「{thread_id}」累计消息 {total} 条，本轮审批 {approvals} 次，"
              f"耗时 {time.time() - t0:.1f}s")
        print("─" * 62)
    return {"answer": final, "approvals": approvals, "thread_id": thread_id}


def print_memory(thread_id: str):
    """查看某个线程的记忆（用于验证会话状态确实已落盘）"""
    config = {"configurable": {"thread_id": thread_id}}
    msgs = (graph.get_state(config).values or {}).get("messages") or []
    print(f"🧠 线程「{thread_id}」共 {len(msgs)} 条消息：")
    for m in msgs:
        role = {"system": "系统", "human": "用户", "ai": "Agent", "tool": "工具"}.get(m.type, m.type)
        text = (m.content or "").replace("\n", " ")
        if getattr(m, "tool_calls", None):
            text = f"调用 {[tc['name'] for tc in m.tool_calls]}"
        print(f"   [{role}] {text[:90]}")


# ════════════════════════════════════════════════════════════════
# 4. 入口
# ════════════════════════════════════════════════════════════════
DEMO_QUESTION = "公司没有任何理由就把我辞退了，我要申请劳动仲裁，帮我写一份申请书。"
DEMO_ANSWERS = ["张三", "某某科技有限公司", "2年半", "9000"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="default", help="会话线程 ID（同一 ID 共享记忆）")
    parser.add_argument("--question", help="只问一个问题（脚本化）")
    parser.add_argument("--demo", action="store_true",
                        help="脚本演示：报案情 + 生成文书（需批准）+ 追问时效（检验记忆）")
    parser.add_argument("--approve", choices=["yes", "no"], default="yes",
                        help="--demo 模式下的自动审批决定")
    parser.add_argument("--show-memory", action="store_true", help="打印该线程的记忆内容")
    parser.add_argument("--trace", action="store_true",
                        help="开启可观测性：采集 LLM/工具调用、token 与成本，结束时输出报告")
    args = parser.parse_args()

    print("=" * 62)
    print("  劳动维权咨询 Agent v3 —— 多轮记忆（Sqlite 检查点）+ 人在环审批")
    print("=" * 62)

    if args.show_memory:
        print_memory(args.thread)
        return

    if not API_KEY:
        print("\n❌ 未找到 OPENAI_API_KEY。请在 .env 中配置。")
        return

    global SCRIPTED_ANSWERS

    collector = TraceCollector(model=MODEL) if args.trace else None
    cbs = [collector] if collector else None
    if collector:
        print(f"📈 可观测性已开启：单价口径 {collector.model}，"
              + ("（当前为高峰计价时段）" if collector.peak else "（当前为非高峰时段）"))

    if args.demo:
        SCRIPTED_ANSWERS = list(DEMO_ANSWERS)
        if args.approve == "yes":
            scripted_approvals = [{"approved": True}]
        else:
            # 否决一次（附修改意见）→ 模型按意见重做 → 第二次批准：
            # 完整演示"否决—修订—通过"链路，也避免队列用尽落到默认值
            scripted_approvals = [
                {"approved": False, "feedback": "赔偿金额计算依据再补充一下"},
                {"approved": True},
            ]
        run_turn(args.thread, DEMO_QUESTION, scripted_approvals=scripted_approvals, callbacks=cbs)
        # 第二轮：换问题但不重述案情，检验记忆是否生效
        print("\n\n" + "=" * 62)
        print("  🧠 记忆检验：第二轮换问题，不重述案情")
        print("=" * 62)
        SCRIPTED_ANSWERS = []
        run_turn(args.thread, "那仲裁时效是多久？我这种情况还来得及吗？", callbacks=cbs)
        if collector:
            collector.print_report(f"线程「{args.thread}」本次演示")
            print(f"📄 明细已追加写入 {collector.dump().name}")
        print("\n💡 跨进程记忆验证：另开一次运行，同一 --thread 换问题即可：")
        print(f'   .venv/Scripts/python agent_hitl.py --thread {args.thread} --question "我的工龄是几年？"')
        return

    if args.question:
        run_turn(args.thread, args.question, callbacks=cbs)
        if collector:
            collector.print_report(f"线程「{args.thread}」单次提问")
            print(f"📄 明细已追加写入 {collector.dump().name}")
        return

    # 交互模式
    print(f"\n当前线程「{args.thread}」，可多轮追问；输入 /memory 查看记忆，回车退出。")
    while True:
        try:
            q = input("\n👤 你：").strip()
        except EOFError:
            break
        if not q:
            break
        if q == "/memory":
            print_memory(args.thread)
            continue
        run_turn(args.thread, q, callbacks=cbs)
    if collector:
        collector.print_report(f"线程「{args.thread}」本次会话累计")
        print(f"📄 明细已追加写入 {collector.dump().name}")
    print("👋 再见（记忆已保存，下次用同一 --thread 可继续）")


if __name__ == "__main__":
    main()
