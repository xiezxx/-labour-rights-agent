"""
劳动维权咨询 Agent —— AI Agent 应用演示原型
====================================================

一个最小但完整的 Agent 实现，用于演示与讲解：

1. 手写 Agent 循环（ReAct 式：思考 → 工具调用 → 观察 → 再思考），未使用 LangGraph 等框架
2. 原生 Function Calling（OpenAI 兼容协议，DeepSeek / OpenAI 均可）
3. 四个工具：法条检索、赔偿计算、时效查询、反问澄清（Human-in-the-loop）
4. 关键设计：
   - 金额由 Python 计算而非 LLM 心算（避免算术幻觉）
   - 法条引用必须来自检索工具结果（避免编造条文号）
   - 循环步数上限，防止失控死循环
   - 信息不足时主动反问，而不是猜测工龄、工资等关键数字

与毕业论文的关系：
论文系统（thesis-rag-labour-law）是"检索增强生成流水线"——改写→检索→门控→生成，
控制流由代码写死，确定性优先，适合法条查询；
本原型把同一领域任务 Agent 化——由模型自主决定调用哪些工具、调用几次、是否追问，
对应论文"未来工作"中的 Agentic RAG 方向。两者互为印证：
流水线保证法律问答的可控性，Agent 展示自主决策与工具编排能力。

运行：
    pip install openai python-dotenv
    python labour_agent.py            # 交互模式
    python labour_agent.py --demo     # 脚本演示（自动应答，适合录屏演示）

配置（.env，与论文项目约定一致）：
    OPENAI_API_KEY=sk-xxx
    OPENAI_BASE_URL=https://api.deepseek.com
    OPENAI_MODEL=deepseek-chat
"""

import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出
    # stdin 同样显式指定，避免管道/IDE 下中文输入按 GBK 解码产生 surrogate 字符
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv

# 优先加载本目录 .env，其次当前工作目录 .env
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env", override=False)
load_dotenv(override=False)

import os
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
MAX_STEPS = 8  # Agent 循环步数上限：防止模型无限调用工具


# ════════════════════════════════════════════════════════════════
# 1. 迷你法条库（演示用，真实条文摘录；生产环境替换为论文项目的混合检索）
# ════════════════════════════════════════════════════════════════
MINI_KB = [
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第10条",
        "content": "建立劳动关系，应当订立书面劳动合同。已建立劳动关系，未同时订立书面劳动合同的，应当自用工之日起一个月内订立书面劳动合同。",
    },
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第14条",
        "content": "用人单位自用工之日起满一年不与劳动者订立书面劳动合同的，视为用人单位与劳动者已订立无固定期限劳动合同。",
    },
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第39条",
        "content": "劳动者有下列情形之一的，用人单位可以解除劳动合同：（一）在试用期间被证明不符合录用条件的；（二）严重违反用人单位的规章制度的；（三）严重失职，营私舞弊，给用人单位造成重大损害的；（四）劳动者同时与其他用人单位建立劳动关系，对完成本单位的工作任务造成严重影响，或者经用人单位提出，拒不改正的；（五）因欺诈、胁迫或乘人之危致使劳动合同无效的；（六）被依法追究刑事责任的。依据本条解除的，用人单位无需支付经济补偿。",
    },
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第40条",
        "content": "有下列情形之一的，用人单位提前三十日以书面形式通知劳动者本人或者额外支付劳动者一个月工资后，可以解除劳动合同：（一）劳动者患病或者非因工负伤，在规定的医疗期满后不能从事原工作，也不能从事由用人单位另行安排的工作的；（二）劳动者不能胜任工作，经过培训或者调整工作岗位，仍不能胜任工作的；（三）劳动合同订立时所依据的客观情况发生重大变化，致使劳动合同无法履行，经用人单位与劳动者协商，未能就变更劳动合同内容达成协议的。",
    },
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第46条",
        "content": "有下列情形之一的，用人单位应当向劳动者支付经济补偿：（一）劳动者依照本法第三十八条规定解除劳动合同的；（二）用人单位依照本法第三十六条规定向劳动者提出解除劳动合同并与劳动者协商一致解除劳动合同的；（三）用人单位依照本法第四十条规定解除劳动合同的；（四）用人单位依照本法第四十一条第一款规定解除劳动合同的；（五）除用人单位维持或者提高劳动合同约定条件续订劳动合同，劳动者不同意续订的情形外，依照本法第四十四条第一项规定终止固定期限劳动合同的；（六）依照本法第四十四条第四项、第五项规定终止劳动合同的；（七）法律、行政法规规定的其他情形。",
    },
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第47条",
        "content": "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。劳动者月工资高于用人单位所在直辖市、设区的市级人民政府公布的本地区上年度职工月平均工资三倍的，向其支付经济补偿的标准按职工月平均工资三倍的数额支付，向其支付经济补偿的年限最高不超过十二年。本条所称月工资是指劳动者在劳动合同解除或者终止前十二个月的平均工资。",
    },
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第48条",
        "content": "用人单位违反本法规定解除或者终止劳动合同，劳动者要求继续履行劳动合同的，用人单位应当继续履行；劳动者不要求继续履行劳动合同或者劳动合同已经不能继续履行的，用人单位应当依照本法第八十七条规定支付赔偿金。",
    },
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第82条",
        "content": "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。用人单位违反本法规定不与劳动者订立无固定期限劳动合同的，自应当订立无固定期限劳动合同之日起向劳动者每月支付二倍的工资。",
    },
    {
        "law": "中华人民共和国劳动合同法",
        "article": "第87条",
        "content": "用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。",
    },
    {
        "law": "中华人民共和国劳动争议调解仲裁法",
        "article": "第27条",
        "content": "劳动争议申请仲裁的时效期间为一年。仲裁时效期间从当事人知道或者应当知道其权利被侵害之日起计算。劳动关系存续期间因拖欠劳动报酬发生争议的，劳动者申请仲裁不受本条第一款规定的仲裁时效期间的限制；但是，劳动关系终止的，应当自劳动关系终止之日起一年内提出。",
    },
    {
        "law": "工资支付暂行规定",
        "article": "第13条",
        "content": "用人单位安排劳动者在日法定标准工作时间以外延长工作时间的，按照不低于劳动合同规定的劳动者本人小时工资标准的150%支付工资；休息日安排工作又不能安排补休的，支付不低于200%的工资报酬；法定休假日安排工作的，支付不低于300%的工资报酬。",
    },
]


# ════════════════════════════════════════════════════════════════
# 2. 工具实现（Python 函数，由 Agent 自主决定何时调用）
# ════════════════════════════════════════════════════════════════
def _bigrams(text: str) -> set:
    """字符二元组分词（演示级轻量检索，无第三方依赖；生产用混合检索）"""
    text = re.sub(r"[\s，。、；：！？（）“”　\-/]", "", text)
    return {text[i : i + 2] for i in range(len(text) - 1)}


def search_law(query: str) -> str:
    """迷你法条库检索：二元组重叠度评分，取 Top3"""
    q_terms = _bigrams(query)
    if not q_terms:
        return "未检索到相关法条，请更换更具体的法律关键词。"
    scored = []
    for doc in MINI_KB:
        d_terms = _bigrams(doc["content"] + doc["article"] + doc["law"])
        if not d_terms:
            continue
        score = len(q_terms & d_terms) / len(q_terms)
        if re.search(r"第\d+条", query):  # 条文号精确命中加分
            m = re.search(r"第(\d+)条", query)
            if m and m.group(1) == re.sub(r"\D", "", doc["article"]):
                score += 100
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return "未检索到相关法条，请更换更具体的法律关键词。"
    hits = []
    for score, doc in scored[:3]:
        hits.append(f"【{doc['law']}{doc['article']}】{doc['content']}（相关度 {score:.2f}）")
    return "\n\n".join(hits)


def _calc_months(years: float) -> float:
    """按《劳动合同法》第47条计算经济补偿月数"""
    y = max(0.0, years)
    if y < 0.5:          # 不满六个月 → 半个月
        return 0.5
    if y < 1.0:          # 六个月以上不满一年 → 按一年
        return 1.0
    full, frac = int(y), y - int(y)
    if frac == 0:
        months = full
    elif frac < 0.5:
        months = full + 0.5
    else:
        months = full + 1.0
    return min(months, 12.0)  # 年限上限 12 年（未计三倍封顶，见 47 条）


def calculate_compensation(years: float, monthly_wage: float, reason: str) -> dict:
    """按《劳动合同法》第47/87/40条计算经济补偿/赔偿金。
    金额由 Python 计算，绝不交给 LLM 心算。"""
    wage = max(0.0, monthly_wage)
    months = _calc_months(years)
    if reason == "过失性辞退":
        total_months, amount, formula, basis = 0.0, 0.0, "0（第39条情形，无经济补偿）", "《劳动合同法》第39条"
    elif reason == "违法解除":
        total_months = months * 2
        amount = round(total_months * wage, 2)
        formula = f"2N = {months} 个月 × 2 = {total_months} 个月工资"
        basis = "《劳动合同法》第87条、第47条"
    elif reason == "无过失性辞退未提前30日通知":
        total_months = months + 1.0
        amount = round(total_months * wage, 2)
        formula = f"N+1 = {months} + 1 = {total_months} 个月工资"
        basis = "《劳动合同法》第40条、第46条、第47条"
    else:  # 无过失性辞退 / 协商解除（单位提出） / 其他经济补偿情形
        total_months = months
        amount = round(total_months * wage, 2)
        formula = f"N = {months} 个月工资"
        basis = "《劳动合同法》第46条、第47条"
    return {
        "工龄_年": years,
        "月工资_元": wage,
        "解除类型": reason,
        "补偿月数": total_months,
        "金额_元": amount,
        "计算公式": formula,
        "法律依据": basis,
        "备注": "未计入社平工资3倍封顶等特殊情形，实际金额以仲裁/法院认定为准",
    }


def check_arbitration_deadline(dispute_type: str) -> str:
    """劳动仲裁时效查询（《劳动争议调解仲裁法》第27条）"""
    if dispute_type == "劳动报酬":
        return (
            "【《劳动争议调解仲裁法》第27条】劳动争议申请仲裁的时效期间为一年。"
            "但劳动关系存续期间因拖欠劳动报酬发生争议的，劳动者申请仲裁不受一年时效期间的限制；"
            "劳动关系终止的，应当自劳动关系终止之日起一年内提出。"
        )
    return (
        "【《劳动争议调解仲裁法》第27条】劳动争议申请仲裁的时效期间为一年，"
        "仲裁时效期间从当事人知道或者应当知道其权利被侵害之日起计算。"
    )


def ask_user(question: str, canned: list = None) -> str:
    """反问澄清：Human-in-the-loop，信息不足时不靠模型猜测"""
    print(f"    ❓ {question}")
    if canned is not None:
        if canned:
            answer = canned.pop(0)
            print(f"    🧑（脚本自动回答）{answer}")
            return answer
        return "（脚本未提供更多回答，请基于已有信息给出结论或建议）"
    try:
        answer = input("    🧑 你的回答：").strip()
    except EOFError:
        return "（无法获取用户输入，请基于已有信息给出结论或建议）"
    return answer or "（用户未回答，请基于已有信息给出结论或建议）"


# ════════════════════════════════════════════════════════════════
# 3. 工具声明（Function Calling 协议）
# ════════════════════════════════════════════════════════════════
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_law",
            "description": "检索劳动法律法规条文。回答中引用的法条必须来自本工具返回结果，严禁凭记忆编造条文号或条文内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "法律检索关键词，如：违法解除劳动合同 赔偿金"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_compensation",
            "description": "按《劳动合同法》第47/87/40条计算经济补偿（N）、代通知金情形（N+1）或违法解除赔偿金（2N）。涉及任何金额都必须调用本工具，严禁自行心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "years": {"type": "number", "description": "工龄（年，支持小数，如 2.5 表示两年半）"},
                    "monthly_wage": {"type": "number", "description": "离职前十二个月平均月工资（元）"},
                    "reason": {
                        "type": "string",
                        "enum": ["违法解除", "无过失性辞退", "无过失性辞退未提前30日通知", "过失性辞退", "协商解除（单位提出）", "其他"],
                        "description": "解除劳动合同的类型，根据案情判断",
                    },
                },
                "required": ["years", "monthly_wage", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_arbitration_deadline",
            "description": "查询劳动仲裁申请时效规则（一年时效及其例外）。用户询问'还能不能要回来''是否过期'等问题时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dispute_type": {
                        "type": "string",
                        "enum": ["劳动报酬", "其他"],
                        "description": "争议类型：拖欠工资等劳动报酬争议填'劳动报酬'，其余填'其他'",
                    },
                },
                "required": ["dispute_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "案情关键信息不足时向用户追问（如工龄、月工资、辞退理由、是否签合同）。一次最多问两个问题，不要猜测关键数字。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要追问用户的问题"},
                },
                "required": ["question"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一名劳动维权咨询 Agent，通过「思考 → 调用工具 → 观察结果 → 再思考」的循环解决问题。

工作流程：
1. 先判断关键信息是否充分（工龄、月工资、辞退理由、是否签合同等）；不足则调用 ask_user 追问，不要猜测；
2. 调用 search_law 检索相关法条；
3. 涉及金额必须调用 calculate_compensation；涉及仲裁时效必须调用 check_arbitration_deadline；
4. 信息齐全后给出最终回答（不要再调用工具）。

最终回答要求：
- 先结论后分析，简洁分点；
- 引用法条写明《法律名》第X条，条文内容以检索结果为准；
- 金额给出计算公式（月数 × 月工资）；
- 结尾提醒：仅供参考，不构成正式法律意见，建议咨询律师或申请劳动仲裁。"""


# ════════════════════════════════════════════════════════════════
# 4. Agent 循环（核心：决策权在模型手里，代码只提供工具和护栏）
# ════════════════════════════════════════════════════════════════
def execute_tool(name: str, args: dict, canned: list = None) -> str:
    if name == "search_law":
        return search_law(str(args.get("query", "")))
    if name == "calculate_compensation":
        try:
            result = calculate_compensation(
                float(args.get("years", 0)),
                float(args.get("monthly_wage", 0)),
                str(args.get("reason", "其他")),
            )
        except (TypeError, ValueError):
            return json.dumps({"error": "参数无效：years/monthly_wage 必须是数字"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)
    if name == "check_arbitration_deadline":
        return check_arbitration_deadline(str(args.get("dispute_type", "其他")))
    if name == "ask_user":
        return ask_user(str(args.get("question", "请补充信息")), canned)
    return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)


def run_agent(question: str, messages: list = None, canned: list = None) -> tuple:
    """一次 Agent 运行：从用户问题出发，循环直到给出最终回答或达到步数上限"""
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": question})
    print(f"\n👤 用户：{question}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    stats = {"steps": 0, "tool_calls": []}

    for step in range(1, MAX_STEPS + 1):
        print(f"  ⏳ 思考中…（第 {step} 步）")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SPECS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # 模型决定不再调用工具 → 给出最终回答，循环结束
            messages.append({"role": "assistant", "content": msg.content})
            print("\n🤖 回答：\n" + (msg.content or "（无内容）"))
            stats["steps"] = step
            return messages, stats

        # 把带 tool_calls 的 assistant 消息原样放回上下文
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  🔧 调用工具：{name}({json.dumps(args, ensure_ascii=False)})")
            result = execute_tool(name, args, canned)
            preview = result[:150].replace("\n", " ")
            print(f"  📥 工具结果：{preview}{'…' if len(result) > 150 else ''}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            stats["tool_calls"].append(name)

    # 步数护栏：防止模型无限调用工具
    print("\n⚠️ 达到最大步数限制（8 步），循环被强制终止——这是 Agent 防失控的护栏设计。")
    stats["steps"] = MAX_STEPS
    return messages, stats


def print_stats(stats: dict):
    print("\n" + "─" * 50)
    print(f"📊 本次运行：{stats['steps']} 步思考，{len(stats['tool_calls'])} 次工具调用")
    if stats["tool_calls"]:
        names = ", ".join(stats["tool_calls"])
        print(f"   调用序列：{names}")
    print("   （这些决策全部由模型在循环中自主做出，代码只提供工具与护栏）")
    print("─" * 50)


# ════════════════════════════════════════════════════════════════
# 5. 入口：--demo 脚本演示 / 交互模式
# ════════════════════════════════════════════════════════════════
DEMO_SCENARIOS = [
    {
        "title": "场景一：违法辞退索赔（反问澄清 + 金额计算）",
        "question": "我被公司突然辞退了，能拿多少赔偿？",
        "answers": [
            "公司没给任何理由，直接让我第二天走人",
            "2年半",
            "8000",
        ],
    },
    {
        "title": "场景二：拖欠工资仲裁时效（时效工具）",
        "question": "公司拖欠我3年工资，我还能要回来吗？",
        "answers": [],
    },
]


def main():
    print("=" * 60)
    print("  劳动维权咨询 Agent —— 演示原型")
    print("  手写 ReAct 循环 + Function Calling + 4 工具")
    print("=" * 60)

    if not API_KEY:
        print("\n❌ 未找到 OPENAI_API_KEY。请在 .env 中配置（见文件头部说明）。")
        return

    if "--demo" in sys.argv:
        for i, scenario in enumerate(DEMO_SCENARIOS, 1):
            print(f"\n\n{'=' * 60}\n🎬 {scenario['title']}\n{'=' * 60}")
            canned = list(scenario["answers"])
            messages, stats = run_agent(scenario["question"], canned=canned)
            print_stats(stats)
        print("\n🎬 演示结束。交互模式请运行：python labour_agent.py")
        return

    messages = None
    print("\n直接输入你的劳动法问题（如：公司没跟我签合同，能要双倍工资吗？）")
    print("可以多轮追问，直接回车退出。\n")
    while True:
        try:
            question = input("👤 你：").strip()
        except EOFError:
            break
        if not question:
            break
        messages, stats = run_agent(question, messages=messages)
        print_stats(stats)
    print("👋 再见")


if __name__ == "__main__":
    main()
