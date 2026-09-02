"""Prompt 缓存命中率本地监控与估算模块（内存版）。

背景
----
接入的第三方 API（如 Agnes、智谱 GLM）大多不返回标准 Anthropic 协议的
cache_read_input_tokens / cache_creation_input_tokens 字段，无法获知服务端
真实的 prompt 缓存命中情况。本模块在 Agent 进程内记录发送给每个 LLM 的
请求原文（messages 序列化文本，剔除图片等二进制内容），并用「连续两条
Prompt 的公共前缀长度」近似模拟服务端 prompt 缓存行为。

估算方法（与产品约定一致）
----
对同一 LLM 的连续两条 Prompt（第 N-1 条与第 N 条）：
- 从第 1 个字符开始逐字符比较，找到第一个不同的位置 d；
- 前 [0, d) 段计为缓存命中（Cache Hit）；
- 从 d 到本条结束计为缓存未命中（Cache Miss）；
- 若第 N 条是第 N-1 条的前缀（比较范围内无差异），整条计为命中；
- 每个 LLM 记录到的第一条 Prompt 无前置上下文，整条计为未命中。

内存管理
----
- 全局单例，按模型名分桶存储；每个模型桶内存上限 64MB（按字符数近似）。
- 超出上限按 FIFO 淘汰最早记录的 Prompt，直到回到上限以内。
- 累计统计（请求数/命中/未命中）独立于内存中的记录列表，淘汰不影响统计。

局限
----
- 仅模拟「前缀缓存」，未考虑服务端可能对多轮对话中的非连续性片段
  （如工具结果、RAG 片段）单独缓存；
- 以字符数近似内存与流量，与真实字节数（UTF-8 下中文占 3 字节）有差异；
- 估算只反映「发出去的文本前缀复用比例」，不代表服务端真实扣费口径。

请求 / 响应双向记录
----
- 请求（To 方向）：record()，记录发送给模型的内容，含前缀命中估算；
- 响应（From 方向）：record_response()，记录模型返回的内容（文本 + 工具调用），
  独立存储、独立分页接口，不做前缀估算；两者按模型名归拢。
"""

import json
import os
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


def _env_int(name: str, default: int) -> int:
    """读取环境变量并转换为正整数，解析失败时回退默认值。"""
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# 每个模型在内存中保留 Prompt 原文的字符数上限（默认 64MB，按字符近似）。
MAX_CHARS_PER_MODEL = _env_int("BIGCODEX_CACHE_MONITOR_MAX_CHARS", 64 * 1024 * 1024)


def _common_prefix_len(prev: str, cur: str) -> int:
    """返回 cur 相对 prev 的公共前缀字符数（即估算的缓存命中字符数）。"""
    n = min(len(prev), len(cur))
    i = 0
    while i < n and prev[i] == cur[i]:
        i += 1
    return i


def _utf16_len(s: str) -> int:
    """UTF-16 码元长度，与前端 JS 字符串的 length/slice 索引一致（代理对计 2）。"""
    return len(s.encode("utf-16-le")) // 2


def _snapshot_path() -> str:
    """临时快照文件路径：位于本模块目录，避免随启动 CWD 变化。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_monitor_snapshot.json")


class PromptRecord:
    """单条 Prompt 记录：原文 + 命中/未命中估算结果。"""

    __slots__ = ("seq", "timestamp", "text", "char_count", "hit_chars", "miss_chars")

    def __init__(self, seq: int, timestamp: int, text: str, hit_chars: int, miss_chars: int) -> None:
        self.seq = seq
        self.timestamp = timestamp
        self.text = text
        self.char_count = _utf16_len(text)
        self.hit_chars = hit_chars
        self.miss_chars = miss_chars

    def to_dict(self) -> Dict[str, Any]:
        """转成接口输出用字典。"""
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "char_count": self.char_count,
            "hit_chars": self.hit_chars,
            "miss_chars": self.miss_chars,
            "text": self.text,
        }


class ResponseRecord:
    """单条 LLM 响应记录（From 方向，无命中估算）。"""

    __slots__ = ("seq", "timestamp", "text", "char_count")

    def __init__(self, seq: int, timestamp: int, text: str) -> None:
        self.seq = seq
        self.timestamp = timestamp
        self.text = text
        self.char_count = _utf16_len(text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "char_count": self.char_count,
            "text": self.text,
        }


class _ModelState:
    """单个模型的监控状态：记录列表 + 累计统计。"""

    def __init__(self) -> None:
        self.records: "deque[PromptRecord]" = deque()
        self.current_chars = 0          # 当前 records 内的总字符数（用于 FIFO 上限判断）
        self.cum_requests = 0           # 累计请求条数
        self.cum_chars = 0              # 累计发送字符数
        self.cum_hit = 0                # 累计估算命中字符数
        self.cum_miss = 0               # 累计估算未命中字符数
        self.responses: "deque[ResponseRecord]" = deque()
        self.current_response_chars = 0  # 当前 responses 内的总字符数
        self.cum_responses = 0           # 累计响应条数


class CacheMonitor:
    """内存版 Prompt 缓存监控器（进程内单例）。"""

    def __init__(self, max_chars_per_model: int = MAX_CHARS_PER_MODEL) -> None:
        self._models: Dict[str, _ModelState] = {}
        self.max_chars_per_model = max_chars_per_model
        self._lock = threading.Lock()
        # 启动时自动预置临时快照（若存在），避免重启后记录为空
        self.load_from_file()

    def record(self, model: str, text: str) -> PromptRecord:
        """记录一条发送给指定模型的 Prompt，返回带命中估算的记录。

        第一条 Prompt 整条计为未命中；后续与上一条比较公共前缀，
        公共前缀计为命中，其余计为未命中。记录后按 FIFO 淘汰超限数据。
        """
        if not model or not text:
            # 空文本不参与估算，但仍计数为一次请求，避免吞掉统计口径
            text = text or ""
        with self._lock:
            state = self._models.setdefault(model, _ModelState())
            state.cum_requests += 1

            prev = state.records[-1].text if state.records else None
            if prev is None:
                hit = 0
            else:
                hit = _common_prefix_len(prev, text)
            # 命中/未命中按 UTF-16 码元长度换算，与前端 JS 字符串索引（length/slice）一致，
            # 避免 Prompt 含 emoji 等非 BMP 字符时前端切分错位。
            total_len = _utf16_len(text)
            hit_len = _utf16_len(text[:hit])
            miss_len = total_len - hit_len

            rec = PromptRecord(
                seq=state.cum_requests,
                timestamp=int(time.time() * 1000),
                text=text,
                hit_chars=hit_len,
                miss_chars=miss_len,
            )
            state.records.append(rec)
            state.current_chars += total_len
            state.cum_chars += total_len
            state.cum_hit += hit_len
            state.cum_miss += miss_len

            # FIFO 淘汰：从最旧的记录开始移除，直到回到上限以内
            while state.current_chars > self.max_chars_per_model and state.records:
                old = state.records.popleft()
                state.current_chars -= old.char_count
            return rec

    def record_response(self, model: str, text: str) -> ResponseRecord:
        """记录一条从指定模型收到的响应内容（From 方向，独立于请求记录）。

        不做前缀命中估算；超出内存上限按 FIFO 淘汰最早响应。
        """
        text = text or ""
        with self._lock:
            state = self._models.setdefault(model, _ModelState())
            state.cum_responses += 1
            rec = ResponseRecord(
                seq=state.cum_responses,
                timestamp=int(time.time() * 1000),
                text=text,
            )
            state.responses.append(rec)
            state.current_response_chars += rec.char_count
            while state.current_response_chars > self.max_chars_per_model and state.responses:
                old = state.responses.popleft()
                state.current_response_chars -= old.char_count
            return rec

    def stats(self) -> List[Dict[str, Any]]:
        """返回各模型的累计统计（自服务启动以来，含已淘汰记录）。"""
        with self._lock:
            out: List[Dict[str, Any]] = []
            for model, st in self._models.items():
                rate = (st.cum_hit / st.cum_chars * 100.0) if st.cum_chars else 0.0
                out.append({
                    "model": model,
                    "requests": st.cum_requests,
                    "total_chars": st.cum_chars,
                    "hit_chars": st.cum_hit,
                    "miss_chars": st.cum_miss,
                    "hit_rate": round(rate, 2),
                    "memory_chars": st.current_chars,
                    "memory_mb": round(st.current_chars / (1024.0 * 1024.0), 2),
                    "max_chars": self.max_chars_per_model,
                    "records_count": len(st.records),
                    "responses": st.cum_responses,
                    "responses_count": len(st.responses),
                })
            return out

    def records(self, model: str, page: int = 1, page_size: int = 20) -> Tuple[int, List[Dict[str, Any]]]:
        """分页返回指定模型当前内存中的 Prompt 记录（新记录在后）。

        返回 (总条数, 当前页记录列表)。"""
        with self._lock:
            st = self._models.get(model)
            if st is None:
                return 0, []
            total = len(st.records)
            start = (page - 1) * page_size
            if start < 0 or start >= total:
                return total, []
            all_items = list(st.records)
            items = all_items[start:start + page_size]
            out: List[Dict[str, Any]] = []
            for i, r in enumerate(items):
                d = r.to_dict()
                nxt = all_items[start + i + 1] if start + i + 1 < total else None
                # 后一条记录的未命中起始位置：即后一条与本条 Prompt 公共前缀的长度，
                # 也是本条内容中下一条 Prompt 开始不同的字符位置；无下一条时为 -1。
                d["next_miss_start"] = nxt.hit_chars if nxt is not None else -1
                out.append(d)
            return total, out

    def responses(self, model: str, page: int = 1, page_size: int = 20) -> Tuple[int, List[Dict[str, Any]]]:
        """分页返回指定模型当前内存中的响应记录（新记录在后）。返回 (总条数, 当前页列表)。"""
        with self._lock:
            st = self._models.get(model)
            if st is None:
                return 0, []
            total = len(st.responses)
            start = (page - 1) * page_size
            if start < 0 or start >= total:
                return total, []
            all_items = list(st.responses)
            items = all_items[start:start + page_size]
            return total, [r.to_dict() for r in items]

    def clear(self, model: Optional[str] = None) -> bool:
        """清除指定模型或全部模型的记录与统计。

        返回是否实际清除了数据。"""
        with self._lock:
            if model is None:
                cleared = bool(self._models)
                self._models.clear()
                return cleared
            if model in self._models:
                del self._models[model]
                return True
            return False

    def clear_snapshot_file(self, path: Optional[str] = None) -> bool:
        """删除落盘快照文件（含 .tmp 临时文件）；清空全部时同步清理，避免重启后旧数据回灌。"""
        path = path or _snapshot_path()
        removed = False
        for p in (path, path + ".tmp"):
            try:
                if os.path.exists(p):
                    os.remove(p)
                    removed = True
            except OSError:
                pass
        return removed

    def clear_test(self) -> None:
        """仅供测试：清空全部状态。"""
        self.clear()

    def to_snapshot_dict(self) -> Dict[str, Any]:
        """把当前内存状态（各模型累计统计 + 记录列表）导出为可序列化字典。"""
        with self._lock:
            models: Dict[str, Any] = {}
            for model, st in self._models.items():
                models[model] = {
                    "cum_requests": st.cum_requests,
                    "cum_chars": st.cum_chars,
                    "cum_hit": st.cum_hit,
                    "cum_miss": st.cum_miss,
                    "cum_responses": st.cum_responses,
                    "records": [r.to_dict() for r in st.records],
                    "responses": [r.to_dict() for r in st.responses],
                }
            return {
                "version": 2,
                "saved_at": int(time.time() * 1000),
                "max_chars_per_model": self.max_chars_per_model,
                "models": models,
            }

    def save_to_file(self, path: Optional[str] = None) -> Dict[str, Any]:
        """把当前缓存数据写入临时快照文件（先写临时文件再原子替换）。"""
        path = path or _snapshot_path()
        snapshot = self.to_snapshot_dict()
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return {
            "saved": True,
            "path": path,
            "models": len(snapshot["models"]),
            "records": sum(len(m["records"]) for m in snapshot["models"].values()),
            "responses": sum(len(m.get("responses") or []) for m in snapshot["models"].values()),
        }

    def load_from_file(self, path: Optional[str] = None) -> bool:
        """从临时快照文件预置数据（启动时调用）。

        快照由 save_to_file 生成，命中/未命中已是 UTF-16 码元单位，直接恢复；
        累计统计保留快照中的值（含已淘汰记录），维持重启后的连续性。
        """
        path = path or _snapshot_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False
        with self._lock:
            self._models.clear()
            loaded = 0
            for model, m in data.get("models", {}).items():
                st = _ModelState()
                st.cum_requests = int(m.get("cum_requests", 0))
                st.cum_chars = int(m.get("cum_chars", 0))
                st.cum_hit = int(m.get("cum_hit", 0))
                st.cum_miss = int(m.get("cum_miss", 0))
                st.cum_responses = int(m.get("cum_responses", 0))
                for rd in m.get("records", []):
                    text = rd.get("text") or ""
                    rec = PromptRecord(
                        seq=int(rd.get("seq", 0)),
                        timestamp=int(rd.get("timestamp", 0)),
                        text=text,
                        hit_chars=max(0, int(rd.get("hit_chars", 0))),
                        miss_chars=max(0, int(rd.get("miss_chars", 0))),
                    )
                    st.records.append(rec)
                    st.current_chars += rec.char_count
                for rd in m.get("responses", []):
                    text = rd.get("text") or ""
                    rrec = ResponseRecord(
                        seq=int(rd.get("seq", 0)),
                        timestamp=int(rd.get("timestamp", 0)),
                        text=text,
                    )
                    st.responses.append(rrec)
                    st.current_response_chars += rrec.char_count
                self._models[model] = st
                loaded += 1
            return loaded > 0


# 进程内全局单例，engine/ 通过它记录请求
CACHE_MONITOR = CacheMonitor()
