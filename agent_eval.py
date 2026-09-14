"""
Agent 评测脚本 —— Agent 版 vs 流水线版（对照实验）
=====================================================

为什么做这个：Agent 的评测是行业公认难题（行为不确定、路径不唯一）。
本脚本刻意**不用 LLM-as-judge**，而是设计一组可自动验证的客观指标，
对同一批案情分别跑「Agent 版（本文具调用循环）」和「流水线版（检索一次→生成）」：

  1. 工具召回率   —— 该调用的工具是否都调用了（如金额问题必须调 calculate_compensation）
  2. 反问行为正确 —— 信息不全时是否主动追问；信息齐全时是否不啰嗦（Agent 版专有能力）
  3. 金额正确率   —— 答案中的赔偿金额是否等于程序计算值（检验"不让 LLM 心算"的价值）
  4. 引用可验证率 —— 答案引用的每个第X条是否出现在本轮检索结果中（检验防幻觉机制）
  5. 耗时与调用次数

公平性设置：流水线版拿到与 Agent 版**相同的信息**（脚本把用户补充信息一次性喂给它），
它的劣势不是"信息更少"，而是"无法边问边查"：只检索一次、不做程序化计算、无白名单约束。

运行：
    .venv/Scripts/python agent_eval.py            # 跑全部用例
    .venv/Scripts/python agent_eval.py --case C01 # 只跑单个用例
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import langgraph_agent as lg
from labour_agent import SYSTEM_PROMPT, search_law as _search_law

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "eval_results.json"

# ════════════════════════════════════════════════════════════════
# 1. 测试集：11 个标准案情（覆盖各类工具路径与边界）
# ════════════════════════════════════════════════════════════════
CASES = [
    {
        "id": "C01",
        "note": "违法辞退 + 信息不全：应反问后计算 2N",
        "question": "我被公司突然辞退了，能拿多少赔偿？",
        "scripted_answers": ["公司没给任何理由，直接让我第二天走人", "2年半", "8000"],
        "required_tools": ["search_law", "calculate_compensation"],
        "expect_clarify": True,
        "expect_amount": 48000.0,
        "expected_articles": ["第87条", "第47条"],
    },
    {
        "id": "C02",
        "note": "违法辞退 + 信息齐全：不应反问，直接算 2N=60000",
        "question": "我在公司干了3年，月薪1万元，昨天公司没有任何理由就把我辞退了，我能拿多少赔偿金？",
        "scripted_answers": [],
        "required_tools": ["search_law", "calculate_compensation"],
        "expect_clarify": False,
        "expect_amount": 60000.0,
        "expected_articles": ["第87条"],
    },
    {
        "id": "C03",
        "note": "拖欠工资时效：必须调时效工具",
        "question": "公司拖欠我3年工资，我还能要回来吗？",
        "scripted_answers": ["我还在职"],
        "required_tools": ["check_arbitration_deadline"],
        "expect_clarify": None,
        "expect_amount": None,
        "expected_articles": ["第27条"],
    },
    {
        "id": "C04",
        "note": "未签书面合同双倍工资",
        "question": "公司一直没跟我签书面劳动合同，我能要双倍工资吗？",
        "scripted_answers": ["入职8个月了，一直没签"],
        "required_tools": ["search_law"],
        "expect_clarify": True,
        "expect_amount": None,
        "expected_articles": ["第82条"],
    },
    {
        "id": "C05",
        "note": "主动辞职是否有补偿",
        "question": "我想主动辞职，能拿到经济补偿吗？",
        "scripted_answers": [],
        "required_tools": ["search_law"],
        "expect_clarify": None,
        "expect_amount": None,
        "expected_articles": ["第46条"],
    },
    {
        "id": "C06",
        "note": "加班费标准（休息日 200%）",
        "question": "我周末经常加班，公司不给加班费，合法吗？",
        "scripted_answers": [],
        "required_tools": ["search_law"],
        "expect_clarify": None,
        "expect_amount": None,
        "expected_articles": ["第13条"],
    },
    {
        "id": "C07",
        "note": "试用期以不符合录用条件辞退（第39条，无需补偿）",
        "question": "试用期第2个月，公司说我不符合录用条件把我辞退了，有补偿吗？",
        "scripted_answers": [],
        "required_tools": ["search_law"],
        "expect_clarify": None,
        "expect_amount": None,
        "expected_articles": ["第39条"],
    },
    {
        "id": "C08",
        "note": "无过失性辞退 + 未提前通知：N+1 = 2.5×9000 = 22500",
        "question": "公司说岗位取消让我走，我干了1年4个月，月薪9000，能拿N+1吗？",
        "scripted_answers": [],
        "required_tools": ["search_law", "calculate_compensation"],
        "expect_clarify": False,
        "expect_amount": 22500.0,
        "expected_articles": ["第40条"],
    },
    {
        "id": "C09",
        "note": "协商解除（单位提出）应有经济补偿",
        "question": "公司跟我协商解除劳动合同，我签了字，还能要经济补偿吗？",
        "scripted_answers": [],
        "required_tools": ["search_law"],
        "expect_clarify": None,
        "expect_amount": None,
        "expected_articles": ["第46条"],
    },
    {
        "id": "C10",
        "note": "知识边界：迷你库无工伤条例，检验是否编造条文号",
        "question": "我在工作中受伤了，算工伤吗？能赔多少？",
        "scripted_answers": [],
        "required_tools": ["search_law"],
        "expect_clarify": None,
        "expect_amount": None,
        "expected_articles": [],
        "kb_boundary": True,
    },
    {
        "id": "C11",
        "note": "工龄含零头（7年8个月）：满6个月不满1年按1年 → N=8个月，2N=240000（LLM 心算易错点）",
        "question": "我工作了7年8个月，月薪15000元，公司违法辞退我，能拿多少赔偿金？",
        "scripted_answers": [],
        "required_tools": ["search_law", "calculate_compensation"],
        "expect_clarify": False,
        "expect_amount": 240000.0,
        "expected_articles": ["第87条"],
    },
]

# ════════════════════════════════════════════════════════════════
# 2. 流水线基线：检索一次 → 单次 LLM 生成（无工具、无反问）
# ════════════════════════════════════════════════════════════════
PIPELINE_PROMPT = """你是一名劳动法律师，请基于以下检索到的法律资料回答用户问题。

## 检索到的法律资料
{context}

## 用户补充信息
{facts}

## 用户问题
{question}

要求：先结论后分析，引用法条写明《法律名》第X条，涉及金额给出计算公式，结尾提醒仅供参考。"""

_PIPELINE_LLM = None


def _pipeline_llm():
    """懒加载流水线基线用的 LLM。

    不在 import 时构造，这样 `--rescore`（纯离线重算指标）无需配置 API Key 也能用，
    全新 clone 也不会因为缺 .env 而连模块都导不进来。
    """
    global _PIPELINE_LLM
    if _PIPELINE_LLM is None:
        _PIPELINE_LLM = ChatOpenAI(
            model=lg.MODEL, api_key=lg.API_KEY, base_url=lg.BASE_URL, temperature=0.2
        )
    return _PIPELINE_LLM


def run_pipeline(case: dict) -> dict:
    """流水线版：一次性检索 + 一次性生成（与论文系统同构，只是检索更简单）"""
    t0 = time.time()
    context = _search_law(case["question"])
    facts = "；".join(case["scripted_answers"]) if case["scripted_answers"] else "（无）"
    resp = _pipeline_llm().invoke(PIPELINE_PROMPT.format(
        context=context, facts=facts, question=case["question"]
    ))
    return {
        "answer": resp.content or "（无内容）",
        "tool_calls": ["search_law"],
        "tool_outputs": [context],
        "latency": time.time() - t0,
        "recursion_hit": False,
    }


def run_agent(case: dict) -> dict:
    """Agent 版：LangGraph 循环，静默执行"""
    lg.VERBOSE = False
    lg.SCRIPTED_ANSWERS = list(case["scripted_answers"])
    _messages, stats = lg.run_turn(
        [SystemMessage(content=SYSTEM_PROMPT)], case["question"], verbose=False
    )
    return stats


# ════════════════════════════════════════════════════════════════
# 3. 自动打分（全部客观可验证，不用 LLM 评分）
# ════════════════════════════════════════════════════════════════
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn2int(s: str):
    """中文数字转整数：八十七→87、四十四→44、一百零七→107、十→10"""
    if not s:
        return None
    if "百" in s:
        head, _, tail = s.partition("百")
        hundreds = _CN_DIGITS.get(head, 1 if head == "" else 0) * 100
        return hundreds + (_cn2int(tail) or 0 if tail else 0)
    if "十" in s:
        head, _, tail = s.partition("十")
        tens = _CN_DIGITS.get(head, 1) if head else 1
        ones = _CN_DIGITS.get(tail, 0) if tail else 0
        return tens * 10 + ones
    if len(s) == 1:
        return _CN_DIGITS.get(s)
    return None


_NUM_PAT = r"(\d+|[零一二两三四五六七八九十百]+)"


def _extract_citations(answer: str) -> list:
    """抽取法条引用，规范化为 (法律名或None, 条号字符串)；兼容中文数字条号"""
    cites = []
    for law, num in re.findall(r"《([^》]{2,30})》\s*第\s*" + _NUM_PAT + r"\s*条", answer):
        n = num if num.isdigit() else _cn2int(num)
        if n is not None:
            cites.append((law, str(n)))
    # 裸条号（前面不是书名号或空白），避免与带法律名的引用重复计数
    for num in re.findall(r"(?<![》\s])第\s*" + _NUM_PAT + r"\s*条", answer):
        n = num if num.isdigit() else _cn2int(num)
        if n is not None:
            cites.append((None, str(n)))
    return cites


def _norm_law(law: str) -> str:
    """法律名规范化：去掉"中华人民共和国"前缀"""
    return re.sub(r"^中华人民共和国", "", (law or "").strip())


def _kb_whitelist() -> dict:
    """知识库白名单：{规范法律名: {条号}}。

    用知识库全集而非单轮检索结果作为白名单——衡量的是"引用能否回溯到知识库"，
    避免"检索为空则任何引用都算无效"的失真。
    """
    from labour_agent import MINI_KB

    wl = {}
    for doc in MINI_KB:
        law = _norm_law(doc["law"])
        num = re.sub(r"\D", "", doc["article"])
        wl.setdefault(law, set()).add(num)
    return wl


_KB = _kb_whitelist()
_KB_ALL_NUMS = {n for nums in _KB.values() for n in nums}


def _extract_amounts(text: str) -> list:
    """抽取答案中的金额，支持"60000元"与"6万元"两种写法"""
    out = []
    for value, unit in re.findall(r"(\d[\d,]*\.?\d*)\s*(万元|万|元)?", text):
        try:
            v = float(value.replace(",", ""))
        except ValueError:
            continue
        if unit in ("万", "万元"):
            v *= 10000
        out.append(v)
    return out


def score(case: dict, result: dict) -> dict:
    answer = result["answer"]
    called = result["tool_calls"]

    # 指标1：工具召回率
    required = case.get("required_tools") or []
    hit = [t for t in required if t in called]
    tool_recall = len(hit) / len(required) if required else 1.0

    # 指标2：反问行为（None = 该用例不评分）
    expect_clarify = case.get("expect_clarify")
    asked = "ask_user" in called
    clarify_ok = None if expect_clarify is None else (asked == expect_clarify)

    # 指标3：金额正确性（支持"60000元"与"6万元"）
    expect_amount = case.get("expect_amount")
    amount_ok = None
    if expect_amount is not None:
        amount_ok = any(abs(n - expect_amount) <= 1.0 for n in _extract_amounts(answer))

    # 指标4：引用可验证率（引用能否回溯到知识库；兼容中文数字条号）
    cites = _extract_citations(answer)
    valid = 0
    for law, art in cites:
        if law:
            # 带法律名：必须该法律名下确有该条
            if art in _KB.get(_norm_law(law), set()):
                valid += 1
        else:
            # 裸条号：知识库任一法律下有该条即可
            if art in _KB_ALL_NUMS:
                valid += 1
    cite_rate = valid / len(cites) if cites else 0.0

    # 指标5：期望法条命中（参考答案中该出现的关键条文）
    expected_articles = case.get("expected_articles") or []
    art_hit = [a for a in expected_articles if a.replace("第", "").replace("条", "") in
               [re.sub(r"\D", "", c[1]) for c in cites]]
    art_recall = len(art_hit) / len(expected_articles) if expected_articles else None

    return {
        "tool_recall": tool_recall,
        "clarify_ok": clarify_ok,
        "amount_ok": amount_ok,
        "cite_rate": cite_rate,
        "cite_total": len(cites),
        "art_recall": art_recall,
        "latency": result["latency"],
        "n_tools": len(called),
    }


# ════════════════════════════════════════════════════════════════
# 4. 主流程与报表
# ════════════════════════════════════════════════════════════════
def fmt(value, is_pct=True, width=8):
    if value is None:
        return "—".center(width)
    if isinstance(value, bool):
        return ("✅" if value else "❌").center(width)
    if is_pct:
        return f"{value * 100:.0f}%".center(width)
    return f"{value:.1f}".center(width)


def aggregate(rows: list) -> dict:
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    bool_vals = lambda key: [r[key] for r in rows if r[key] is not None]
    return {
        "tool_recall": avg("tool_recall"),
        "clarify_ok": (sum(bool_vals("clarify_ok")) / len(bool_vals("clarify_ok"))
                       if bool_vals("clarify_ok") else None),
        "amount_ok": (sum(bool_vals("amount_ok")) / len(bool_vals("amount_ok"))
                      if bool_vals("amount_ok") else None),
        "cite_rate": avg("cite_rate"),
        "art_recall": avg("art_recall"),
        "latency": avg("latency"),
    }


def print_report(all_results: list):
    """打印汇总表与关键差异"""
    print("\n" + "=" * 78)
    print("  汇总（工具召回 / 反问正确 / 金额正确 / 引用可验证 / 关键条文命中 / 耗时）")
    print("=" * 78)
    print(f"{'用例':<6}{'架构':<10}{'工具召回':>10}{'反问正确':>10}{'金额正确':>10}"
          f"{'引用可验证':>12}{'条文命中':>10}{'耗时(s)':>9}")
    print("-" * 78)
    for r in all_results:
        for mode, key in (("Agent", "agent"), ("流水线", "pipeline")):
            s = r[key]
            print(f"{r['case']['id']:<6}{mode:<10}"
                  f"{fmt(s['tool_recall']):>10}{fmt(s['clarify_ok']):>10}"
                  f"{fmt(s['amount_ok']):>10}{fmt(s['cite_rate']):>12}"
                  f"{fmt(s['art_recall']):>10}{s['latency']:>9.1f}")
    print("-" * 78)
    agg_agent = aggregate([r["agent"] for r in all_results])
    agg_pipe = aggregate([r["pipeline"] for r in all_results])
    for mode, agg in (("Agent", agg_agent), ("流水线", agg_pipe)):
        print(f"{'平均':<6}{mode:<10}"
              f"{fmt(agg['tool_recall']):>10}{fmt(agg['clarify_ok']):>10}"
              f"{fmt(agg['amount_ok']):>10}{fmt(agg['cite_rate']):>12}"
              f"{fmt(agg['art_recall']):>10}{agg['latency']:>9.1f}")
    print("=" * 78)

    print("\n关键差异：")
    print(f"  · 反问澄清：Agent {fmt(agg_agent['clarify_ok']).strip()} vs 流水线 "
          f"{fmt(agg_pipe['clarify_ok']).strip()} —— 流水线架构上不具备追问能力")
    print(f"  · 引用可验证率：Agent {fmt(agg_agent['cite_rate']).strip()} vs 流水线 "
          f"{fmt(agg_pipe['cite_rate']).strip()} —— 引用能否回溯到知识库")
    print(f"  · 金额正确率：Agent {fmt(agg_agent['amount_ok']).strip()} vs 流水线 "
          f"{fmt(agg_pipe['amount_ok']).strip()} —— 程序计算 vs LLM 心算")
    print(f"  · 工具召回率：Agent {fmt(agg_agent['tool_recall']).strip()} vs 流水线 "
          f"{fmt(agg_pipe['tool_recall']).strip()} —— 流水线只有检索一步，无工具选择能力（结构上限）")
    print(f"  · 平均耗时：Agent {agg_agent['latency']:.1f}s vs 流水线 {agg_pipe['latency']:.1f}s")
    print("    （单次采样受模型服务波动影响大；Agent 的结构性成本是更多 LLM 往返次数）")
    print("    → 生产建议：按问题类型路由——简单法条查询走流水线（低延迟、可控），"
          "复杂案情诊断走 Agent（可追问、可计算、引用可验证）")
    return agg_agent, agg_pipe


def rescore_stored(results: list) -> list:
    """离线重算指标（不调用 LLM）。

    金额/引用/条文命中可从答案文本重算；工具召回与反问行为依赖工具调用序列
    （旧结果未存原始序列），沿用已存的原始判定不变。
    """
    for r in results:
        for key in ("agent", "pipeline"):
            old = r[key]
            fresh = score(r["case"], {
                "answer": r.get(f"{key}_answer", ""),
                "tool_calls": [],
                "latency": old["latency"],
            })
            fresh["tool_recall"] = old["tool_recall"]
            fresh["clarify_ok"] = old["clarify_ok"]
            fresh["latency"] = old["latency"]
            fresh["n_tools"] = old["n_tools"]
            r[key] = fresh
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="只跑指定用例 ID，如 C01")
    parser.add_argument("--merge", action="store_true",
                        help="与已有 eval_results.json 合并（新增用例时用）")
    parser.add_argument("--rescore", action="store_true",
                        help="离线重算指标（不调用 LLM），用于打分口径修正")
    args = parser.parse_args()

    if args.rescore:
        if not RESULTS_PATH.exists():
            print("未找到 eval_results.json")
            return
        stored = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        stored = rescore_stored(stored)
        RESULTS_PATH.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已用修正后的打分口径重算 {len(stored)} 个用例，未调用 LLM。\n")
        print_report(stored)
        return

    cases = [c for c in CASES if not args.case or c["id"] == args.case]
    if not cases:
        print(f"未找到用例 {args.case}")
        return
    if not lg.API_KEY:
        print("❌ 未找到 OPENAI_API_KEY")
        return

    print("=" * 78)
    print(f"  Agent 评测：{len(cases)} 个标准案情 × 2 种架构（Agent 版 vs 流水线版）")
    print("  指标全部自动化客观验证，未使用 LLM-as-judge")
    print("=" * 78)

    new_results = []
    for case in cases:
        print(f"\n▶ {case['id']} {case['note']}")
        print(f"  Q: {case['question']}")

        print("  [Agent 版] 运行中…", end="", flush=True)
        agent_result = run_agent(case)
        agent_score = score(case, agent_result)
        print(f"\r  [Agent 版] 工具={agent_result['tool_calls']} "
              f"耗时={agent_score['latency']:.1f}s        ")

        print("  [流水线版] 运行中…", end="", flush=True)
        pipe_result = run_pipeline(case)
        pipe_score = score(case, pipe_result)
        print(f"\r  [流水线版] 工具={pipe_result['tool_calls']} "
              f"耗时={pipe_score['latency']:.1f}s        ")

        new_results.append({
            "case": case, "agent": agent_score, "pipeline": pipe_score,
            "agent_answer": agent_result["answer"],
            "pipeline_answer": pipe_result["answer"],
            "agent_tools": agent_result["tool_calls"],
            "pipeline_tools": pipe_result["tool_calls"],
        })

    if args.merge and RESULTS_PATH.exists():
        stored = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        by_id = {r["case"]["id"]: r for r in stored}
        for r in new_results:
            by_id[r["case"]["id"]] = r
        all_results = sorted(by_id.values(), key=lambda r: r["case"]["id"])
    else:
        all_results = new_results

    print_report(all_results)
    RESULTS_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 明细已写入 {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
