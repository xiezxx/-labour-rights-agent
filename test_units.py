"""
不依赖 LLM / 网络的单元测试
==============================

覆盖那些"出错了不容易发现"的纯逻辑：悬空工具调用修补、中文数字解析、法条引用抽取、
金额抽取、赔偿月数取整、成本计算。全部离线可跑（不花一分钱）：

    .venv/Scripts/python test_units.py
"""

import io
import sys
import unittest

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent_eval as ae
from labour_agent import MINI_KB, _calc_months
from observability import TraceCollector


def _ai_with_calls(ids, name="search_law"):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": {"query": "x"}, "id": i} for i in ids
    ])


class TestRepairDanglingToolCalls(unittest.TestCase):
    """悬空 tool_call 修补：补的回执必须紧跟 tool_calls 消息，不能追加到末尾"""

    def setUp(self):
        # 延迟导入：导入 agent_hitl 会编译图并打开 sqlite
        from agent_hitl import repair_dangling_tool_calls
        self.repair = repair_dangling_tool_calls

    def test_no_dangling_returns_unchanged_order(self):
        msgs = [_ai_with_calls(["c1"]), ToolMessage(content="ok", tool_call_id="c1")]
        self.assertEqual(self.repair(msgs), msgs)

    def test_dangling_repair_inserted_right_after(self):
        msgs = [_ai_with_calls(["c1"])]
        out = self.repair(msgs)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], ToolMessage)
        self.assertEqual(out[1].tool_call_id, "c1")
        self.assertEqual(out[0], msgs[0])

    def test_dangling_before_human_message_stays_before_it(self):
        """关键回归：回执必须插在 Human 消息之前，否则 API 仍会 400"""
        msgs = [_ai_with_calls(["c1"]), HumanMessage(content="接着问")]
        out = self.repair(msgs)
        self.assertIsInstance(out[1], ToolMessage)
        self.assertIsInstance(out[2], HumanMessage)

    def test_multiple_calls_and_partial_answers(self):
        """已有回执保持在原位，缺失的补在同一段回执之后（不打乱顺序）"""
        msgs = [
            _ai_with_calls(["c1", "c2"]),
            ToolMessage(content="ok", tool_call_id="c1"),
        ]
        out = self.repair(msgs)
        self.assertEqual([type(m).__name__ for m in out],
                         ["AIMessage", "ToolMessage", "ToolMessage"])
        self.assertEqual([out[1].tool_call_id, out[2].tool_call_id], ["c1", "c2"])

    def test_every_tool_call_gets_a_response(self):
        """不变式：修补后每个 tool_call_id 都有回执，且都在下一条非 tool 消息之前"""
        msgs = [
            _ai_with_calls(["a", "b"]),
            HumanMessage(content="问题"),
            _ai_with_calls(["c"]),
        ]
        out = self.repair(msgs)
        seen = set()
        for i, m in enumerate(out):
            if isinstance(m, AIMessage):
                seen = set()
            elif isinstance(m, ToolMessage):
                seen.add(m.tool_call_id)
            else:  # 非 tool 消息：此前的 tool_calls 必须都已配平
                prev_ai = out[i - 1] if i else None
                if isinstance(prev_ai, AIMessage):
                    self.assertFalse(True, "AIMessage 后未紧跟回执")
        self.assertTrue({"a", "b", "c"} <= {m.tool_call_id for m in out
                                            if isinstance(m, ToolMessage)})


class TestChineseNumber(unittest.TestCase):
    def test_values(self):
        for text, expect in [("八十七", 87), ("四十四", 44), ("十", 10), ("十一", 11),
                             ("二十", 20), ("一百零七", 107), ("一百二十", 120),
                             ("一千二百", 1200), ("87", 87)]:
            with self.subTest(text=text):
                self.assertEqual(ae._cn2int(text), expect)

    def test_invalid(self):
        self.assertIsNone(ae._cn2int(""))
        self.assertIsNone(ae._cn2int("甲乙"))


class TestCitationExtraction(unittest.TestCase):
    def _nums(self, text):
        return [int(a) for _law, a in ae._extract_citations(text)]

    def test_bare_article_after_newline_or_bullet(self):
        """回归：旧的位置断言会把换行/项目符号后的裸条号整条漏抽"""
        self.assertEqual(self._nums("依据如下：\n第八十七条 用人单位"), [87])
        self.assertEqual(self._nums("- 第46条明确限定"), [46])
        self.assertEqual(self._nums("详见 第87条 的规定"), [87])

    def test_law_qualified_not_double_counted(self):
        self.assertEqual(self._nums("《劳动合同法》第87条"), [87])
        self.assertEqual(self._nums("《劳动合同法》 第87条"), [87])  # 书名号后有空格
        self.assertEqual(self._nums("《中华人民共和国劳动合同法》第八十七条"), [87])

    def test_no_false_positive(self):
        for text in ["第3个月", "符合第二项", "第三章", "第五款", "2024年"]:
            with self.subTest(text=text):
                self.assertEqual(self._nums(text), [])


class TestAmountExtraction(unittest.TestCase):
    def test_forms(self):
        for text, expect_in in [("6万元", 60000.0), ("22500元", 22500.0),
                                ("54,000.00 元", 54000.0), ("4.8万元", 48000.0)]:
            with self.subTest(text=text):
                self.assertIn(expect_in, ae._extract_amounts(text))


class TestAmountIsClaimed(unittest.TestCase):
    """金额判定：只有以"主张/计算"口吻出现才算答对（旧口径"提及即通过"太松）"""

    def test_negated_amount_not_counted(self):
        self.assertFalse(ae._amount_is_claimed("……只能拿到24000元。注意：不是48000元。", 48000))

    def test_hypothetical_amount_not_counted(self):
        self.assertFalse(ae._amount_is_claimed(
            "公司应支付经济补偿13500元。虽然理论上N+1是22500元，但本案不适用。", 22500))

    def test_claimed_amount_counted(self):
        self.assertTrue(ae._amount_is_claimed("经计算，赔偿金应为240,000元。", 240000))
        self.assertTrue(ae._amount_is_claimed("**赔偿金 2N = 48000 元**", 48000))
        self.assertTrue(ae._amount_is_claimed("可主张赔偿金合计6万元", 60000))

    def test_other_amounts_not_counted(self):
        self.assertFalse(ae._amount_is_claimed("赔偿金为13500元。", 22500))

    def test_bare_number_without_unit_ignored(self):
        self.assertFalse(ae._amount_is_claimed("工龄22500天", 22500))


class TestZeroCitationIsNotApplicable(unittest.TestCase):
    """零引用的正确拒答应记 N/A，而不是 0 分拖低平均"""

    def _score(self, answer, case=None):
        return ae.score(case or {"expected_articles": []},
                        {"answer": answer, "tool_calls": [], "latency": 0.0})

    def test_refusal_scores_none(self):
        self.assertIsNone(self._score("本知识库未收录相关条文，无法给出具体条文号。")["cite_rate"])

    def test_normal_answer_still_scored(self):
        self.assertEqual(self._score("依据《劳动合同法》第87条")["cite_rate"], 1.0)

    def test_aggregate_skips_none(self):
        def row(cite):
            return {"tool_recall": 1.0, "clarify_ok": None, "amount_ok": None,
                    "cite_rate": cite, "art_recall": None, "latency": 1.0}
        self.assertEqual(ae.aggregate([row(1.0), row(None)])["cite_rate"], 1.0)


class TestCompensationMonths(unittest.TestCase):
    """《劳动合同法》第47条的月数取整规则"""

    def test_rounding(self):
        for years, expect in [(0.4, 0.5), (0.5, 1.0), (0.99, 1.0), (1.0, 1.0),
                              (1.4, 1.5), (1.5, 2.0), (2.5, 3.0), (13.0, 12.0)]:
            with self.subTest(years=years):
                self.assertEqual(_calc_months(years), expect)


class TestKnowledgeBase(unittest.TestCase):
    def test_kb_articles_are_arabic(self):
        """白名单靠正则从 article 字段提数字，非阿拉伯数字会让指标失真"""
        import re
        for doc in MINI_KB:
            with self.subTest(article=doc["article"]):
                self.assertEqual(re.sub(r"\D", "", doc["article"]), doc["article"][1:-1])


class TestCostCalculation(unittest.TestCase):
    def test_cost_split(self):
        c = TraceCollector(model="deepseek-v4-pro")
        price = c.price
        cost = c._cost(1_000_000, 1_000_000, 1_000_000)
        expect = price["cache_hit"] + price["cache_miss"] + price["output"]
        self.assertAlmostEqual(cost, expect, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
