"""
Agent 可观测性：本地追踪 + 成本核算（并可选接入 LangSmith）
================================================================

生产环境的 Agent 必须能回答三个问题：**花了多少钱、慢在哪一步、哪次调用失败了**。
本模块提供一个 LangChain 回调处理器（BaseCallbackHandler），零外部依赖、零账号，
把每次运行的可观测数据采集下来：

- 每次 LLM 调用：输入/输出 token（区分缓存命中与未命中）、耗时、估算成本
- 每次工具调用：工具名、参数、耗时、返回长度、异常
- 汇总报告：控制台打印 + 追加写入 trace_log.jsonl

为什么要自己做而不是只用 LangSmith：
1. 不需要账号与外部服务，本地即可用（也是离线演示的前提）
2. 能按自己的口径核算成本（含 DeepSeek 的缓存命中折扣）
3. 理解了回调机制，再上平台只是"多一个 callback"的事

接入 LangSmith（可选，开启后本模块与 LangSmith 会同时生效）：
    # .env
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_xxx        # 从 smith.langchain.com 获取
    LANGSMITH_PROJECT=labour-agent
    HTTPS_PROXY=http://127.0.0.1:7893   # LangSmith 在境外，需代理
    注意：开启后追踪数据会上传到 LangSmith 云端（数据离开本机），自己权衡。

单价口径说明（重要）：
    下方 PRICE_TABLE 是 2026-09 检索到的 DeepSeek 公开价（元/百万 tokens），
    仅用于**成本量级估算**。厂商会调价、也有峰谷分时，请以实际账单为准；
    可用环境变量覆盖，避免把估算当账单。
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文/emoji 输出
except Exception:
    pass

from langchain_core.callbacks import BaseCallbackHandler

SCRIPT_DIR = Path(__file__).resolve().parent
TRACE_LOG = SCRIPT_DIR / "trace_log.jsonl"

# ── 单价表（元 / 百万 tokens）──
# cache_hit：输入缓存命中价；cache_miss：输入未命中价；output：输出价
# peak_multiplier：高峰时段倍率（DeepSeek Flash 系列有峰谷分时计价，pro 未见公开说明，取 1.0）
PRICE_TABLE = {
    "deepseek-v4-pro":   {"cache_hit": 0.025, "cache_miss": 3.0, "output": 6.0, "peak_multiplier": 1.0},
    "deepseek-v4-flash": {"cache_hit": 0.02,  "cache_miss": 1.0, "output": 4.0, "peak_multiplier": 2.0},
    "deepseek-chat":     {"cache_hit": 0.02,  "cache_miss": 1.0, "output": 4.0, "peak_multiplier": 2.0},
}
FALLBACK_PRICE = {"cache_hit": 0.025, "cache_miss": 3.0, "output": 6.0, "peak_multiplier": 1.0}

# 高峰时段（北京时间，周一至周五 9:00-12:00 与 14:00-18:00）
PEAK_WINDOWS = ((9, 12), (14, 18))


def _price_for(model: str) -> dict:
    """取单价，支持用环境变量整体覆盖（LLM_PRICE_CACHE_HIT / _CACHE_MISS / _OUTPUT）"""
    price = dict(PRICE_TABLE.get(model, FALLBACK_PRICE))
    for key, env in (("cache_hit", "LLM_PRICE_CACHE_HIT"),
                     ("cache_miss", "LLM_PRICE_CACHE_MISS"),
                     ("output", "LLM_PRICE_OUTPUT")):
        if os.getenv(env):
            try:
                price[key] = float(os.getenv(env))
            except ValueError:
                pass
    return price


def _is_peak(now: datetime = None) -> bool:
    """是否处于高峰计价时段（北京时间，周一至周五）"""
    now = now or datetime.now(timezone(timedelta(hours=8)))
    if now.weekday() >= 5:  # 周六日无高峰
        return False
    return any(start <= now.hour < end for start, end in PEAK_WINDOWS)


class TraceCollector(BaseCallbackHandler):
    """采集一次 Agent 运行的可观测数据（LLM 调用 / 工具调用 / 异常 / 成本）"""

    def __init__(self, model: str = None, verbose: bool = True):
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-v4-pro")
        self.price = _price_for(self.model)
        self.peak = _is_peak() and self.price.get("peak_multiplier", 1.0) != 1.0
        self.verbose = verbose

        self.llm_events = []
        self.tool_events = []
        self.errors = []
        self._llm_t0 = {}
        self._tool_t0 = {}
        self._tool_meta = {}

    # ── LLM 回调 ──
    def on_llm_start(self, serialized, prompts, run_id=None, **kwargs):
        self._llm_t0[str(run_id)] = time.time()

    def on_llm_end(self, response, run_id=None, **kwargs):
        t0 = self._llm_t0.pop(str(run_id), None)
        latency = (time.time() - t0) if t0 else 0.0
        in_tok, out_tok, cache_hit, cache_miss = self._extract_usage(response)
        cost = self._cost(cache_hit, cache_miss, out_tok)
        self.llm_events.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "latency_s": round(latency, 2),
            "input_tokens": in_tok, "cache_hit_tokens": cache_hit,
            "cache_miss_tokens": cache_miss, "output_tokens": out_tok,
            "cost_cny": round(cost, 6),
        })

    def on_llm_error(self, error, run_id=None, **kwargs):
        self._llm_t0.pop(str(run_id), None)
        self.errors.append({"type": "llm_error", "error": str(error)[:300]})

    # ── 工具回调 ──
    def on_tool_start(self, serialized, input_str, run_id=None, **kwargs):
        rid = str(run_id)
        self._tool_t0[rid] = time.time()
        self._tool_meta[rid] = serialized.get("name") or (serialized.get("id") or ["?"])[-1]

    def on_tool_end(self, output, run_id=None, **kwargs):
        rid = str(run_id)
        t0 = self._tool_t0.pop(rid, None)
        name = self._tool_meta.pop(rid, "?")
        self.tool_events.append({
            "name": name,
            "latency_s": round((time.time() - t0) if t0 else 0.0, 2),
            "result_chars": len(str(output)),
            "error": False,
        })

    def on_tool_error(self, error, run_id=None, **kwargs):
        rid = str(run_id)
        self._tool_t0.pop(rid, None)
        name = self._tool_meta.pop(rid, "?")
        self.tool_events.append({"name": name, "latency_s": 0.0, "result_chars": 0, "error": True})
        self.errors.append({"type": "tool_error", "tool": name, "error": str(error)[:300]})

    # ── 内部工具 ──
    @staticmethod
    def _extract_usage(response) -> tuple:
        """从 LLMResult 里取 token 用量，兼容新式 usage_metadata 与旧式 token_usage。
        取不到缓存命中信息时，保守地按"全部未命中"计价（宁可高估成本）。"""
        in_tok = out_tok = cache_hit = 0
        cache_miss = None
        try:
            message = getattr(response.generations[0][0], "message", None)
            um = getattr(message, "usage_metadata", None) or {}
            in_tok = um.get("input_tokens", 0) or 0
            out_tok = um.get("output_tokens", 0) or 0
            cache_hit = (um.get("input_token_details") or {}).get("cache_read", 0) or 0
        except Exception:
            pass
        if not in_tok:
            tu = (response.llm_output or {}).get("token_usage") or {}
            in_tok = tu.get("prompt_tokens", 0) or 0
            out_tok = tu.get("completion_tokens", 0) or 0
            cache_hit = tu.get("prompt_cache_hit_tokens", 0) or cache_hit
            cache_miss = tu.get("prompt_cache_miss_tokens")
        if cache_miss is None:
            cache_miss = max(0, in_tok - cache_hit)
        return in_tok, out_tok, cache_hit, cache_miss

    def _cost(self, cache_hit: int, cache_miss: int, out_tok: int) -> float:
        mult = self.price.get("peak_multiplier", 1.0) if self.peak else 1.0
        return (cache_hit / 1e6 * self.price["cache_hit"]
                + cache_miss / 1e6 * self.price["cache_miss"]
                + out_tok / 1e6 * self.price["output"]) * mult

    # ── 汇总与输出 ──
    def summary(self) -> dict:
        cost = sum(e["cost_cny"] for e in self.llm_events)
        lat = [e["latency_s"] for e in self.llm_events]
        by_tool = {}
        for e in self.tool_events:
            b = by_tool.setdefault(e["name"], {"count": 0, "total_s": 0.0, "errors": 0})
            b["count"] += 1
            b["total_s"] += e["latency_s"]
            b["errors"] += 1 if e["error"] else 0
        return {
            "model": self.model,
            "price": self.price,
            "peak_applied": self.peak,
            "llm_calls": len(self.llm_events),
            "llm_total_s": round(sum(lat), 2),
            "llm_avg_s": round(sum(lat) / len(lat), 2) if lat else 0.0,
            "llm_max_s": round(max(lat), 2) if lat else 0.0,
            "tool_calls": len(self.tool_events),
            "input_tokens": sum(e["input_tokens"] for e in self.llm_events),
            "cache_hit_tokens": sum(e["cache_hit_tokens"] for e in self.llm_events),
            "cache_miss_tokens": sum(e["cache_miss_tokens"] for e in self.llm_events),
            "output_tokens": sum(e["output_tokens"] for e in self.llm_events),
            "cost_cny": round(cost, 6),
            "errors": len(self.errors),
            "by_tool": by_tool,
        }

    def print_report(self, title: str = "本次运行"):
        s = self.summary()
        price_note = (f"{s['model']}（命中 {s['price']['cache_hit']}/未命中 {s['price']['cache_miss']}"
                      f"/输出 {s['price']['output']} 元每百万 tokens"
                      + ("，已按高峰倍率" if s["peak_applied"] else "") + "）")
        print("\n" + "━" * 62)
        print(f"📈 可观测性报告（{title}）")
        print("━" * 62)
        print(f"  LLM 调用     {s['llm_calls']} 次   累计 {s['llm_total_s']}s   "
              f"平均 {s['llm_avg_s']}s   最慢 {s['llm_max_s']}s")
        print(f"  工具调用     {s['tool_calls']} 次   " + (
            "、".join(f"{k}×{v['count']}" for k, v in s["by_tool"].items()) or "无"))
        print(f"  Token 用量   输入 {s['input_tokens']:,}（缓存命中 {s['cache_hit_tokens']:,} / "
              f"未命中 {s['cache_miss_tokens']:,}）+ 输出 {s['output_tokens']:,}")
        print(f"  估算成本     ¥{s['cost_cny']:.4f}   单价口径：{price_note}")
        print(f"  异常         {s['errors']} 次" + (
            "  " + json.dumps(self.errors[:3], ensure_ascii=False) if self.errors else ""))
        if s["tool_calls"]:
            slow = sorted(self.tool_events, key=lambda e: -e["latency_s"])[:3]
            print("  工具耗时 Top3  " + "、".join(f"{e['name']} {e['latency_s']}s" for e in slow))
        print("  （单价为公开价目估算，请以实际账单为准；可用 LLM_PRICE_* 环境变量覆盖）")
        print("━" * 62)
        return s

    def dump(self, path: Path = None):
        """把本次运行的明细追加写入 jsonl，便于离线分析"""
        path = path or TRACE_LOG
        record = {"summary": self.summary(), "llm_events": self.llm_events,
                  "tool_events": self.tool_events, "errors": self.errors}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path


if __name__ == "__main__":
    # 自检：不调用 LLM，验证计价与汇总逻辑
    c = TraceCollector(model="deepseek-v4-pro", verbose=False)
    c.llm_events = [{"ts": "", "latency_s": 3.1, "input_tokens": 5000,
                     "cache_hit_tokens": 1000, "cache_miss_tokens": 4000,
                     "output_tokens": 800, "cost_cny": c._cost(1000, 4000, 800)}]
    c.tool_events = [{"name": "search_law", "latency_s": 0.01, "result_chars": 300, "error": False}]
    s = c.print_report("自检")
    expect = (1000 / 1e6 * 0.025) + (4000 / 1e6 * 3.0) + (800 / 1e6 * 6.0)
    print(f"自检：成本计算 {'✅ 正确' if abs(s['cost_cny'] - expect) < 1e-9 else '❌ 不符'}"
          f"（期望 ¥{expect:.4f}）")
