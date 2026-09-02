"""LLM 实际往来流量存档与统计（SQLite 持久化）。

背景
----
软件进入实际使用后，需要准确掌握与远端 LLM 之间的实际来往流量（发生时间、
消息大小、模型名称、出入方向、tokens、状态、耗时等），用于成本与使用分析。
本模块在每次真实发往远端 LLM 的 HTTP 交互点埋点，把「除了原文以外的所有属性」
逐条写入 SQLite（默认 ~/.bigcodex/traffic.db，可用 BIGCODEX_TRAFFIC_DB 覆盖）。

口径说明
----
- 一次 LLM HTTP 往返记两条：request（发出时）+ response（收到后）。
- request.bytes = 发送 payload JSON 的 UTF-8 字节数（真实上行流量）；
- response.bytes = 模型输出正文（text+thinking+tool_use）UTF-8 字节数（流式）
  或完整响应 JSON 字节数（非流式）；HTTP 头等固定协议开销不统计。
- 限流/重试：每次实际发出的 HTTP 尝试都记一条 request（429 等失败尝试
  带 status_code 与 error），成功收到响应后再记一条 response，流量如实入账。
- LocalLLM 本地处理器不经过任何远端调用点，天然不记录；
  另外 provider = local 时也做一层过滤保险。
- 原文不落盘，只存属性。

聚合能力
----
柱状图所需的「按桶（小时/天/周）+ 按模型过滤 + 累计字节/次数/tokens」
直接用 SQL GROUP BY 完成，由本模块负责补齐缺失时间桶，返回连续时间轴。
数据量到几十万条仍可秒级返回。
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


_DEFAULT_DB_NAME = "traffic.db"


def _db_path() -> str:
    """流量库文件路径：环境变量 BIGCODEX_TRAFFIC_DB 优先，否则 <data_dir>/traffic.db。

    与 server.py 的 _data_dir() 保持一致（默认 ~/.bigcodex，BIGCODEX_DATA_DIR 可覆盖）；
    独立环境变量便于测试与多实例隔离。
    """
    env = os.environ.get("BIGCODEX_TRAFFIC_DB", "").strip()
    if env:
        return env
    d = os.environ.get("BIGCODEX_DATA_DIR") or os.path.join(os.path.expanduser("~"), ".bigcodex")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, _DEFAULT_DB_NAME)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_traffic (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              INTEGER NOT NULL,              -- 发生时间（epoch 毫秒）
    direction       TEXT    NOT NULL,              -- request（入）| response（出）
    model           TEXT    NOT NULL,              -- 真实模型名
    provider        TEXT    NOT NULL DEFAULT '',   -- 提供商 id（agnes/zhipu-glm/siliconflow/.../local）
    api_format      TEXT    NOT NULL DEFAULT '',   -- openai | anthropic | response
    message_type    TEXT    NOT NULL DEFAULT '',   -- user_turn | tool_round | summarize | retry 等
    session_id      TEXT    NOT NULL DEFAULT '',   -- 会话 id（cli_xxx / desk_xxx）
    stdin_id        TEXT    NOT NULL DEFAULT '',   -- 桥接 stdin id（如有）
    cwd             TEXT    NOT NULL DEFAULT '',   -- 工作目录
    role            TEXT    NOT NULL DEFAULT '',   -- chat | programming
    thinking_level  TEXT    NOT NULL DEFAULT '',   -- off/low/medium/high/max
    stream_supported INTEGER NOT NULL DEFAULT 0,   -- 是否流式（1/0）
    bytes           INTEGER NOT NULL DEFAULT 0,    -- 消息大小（UTF-8 字节；口径见模块 docstring）
    chars           INTEGER NOT NULL DEFAULT 0,    -- 字符数
    input_tokens    INTEGER NOT NULL DEFAULT 0,    -- 输入 tokens（response 侧带网关真实值）
    cache_hit_tokens  INTEGER NOT NULL DEFAULT 0,  -- 命中上下文缓存的输入 token 数（response 侧）
    cache_miss_tokens INTEGER NOT NULL DEFAULT 0,  -- 未命中缓存、按全价计费的输入 token 数（response 侧）
    output_tokens   INTEGER NOT NULL DEFAULT 0,    -- 输出 tokens（response 侧带网关真实值）
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,   -- 模型思考链 tokens（response 侧；缺失记 0）
    status_code     INTEGER NOT NULL DEFAULT 0,    -- HTTP 状态（200 成功 / 429 限流 / 其他）
    duration_ms     INTEGER NOT NULL DEFAULT 0,    -- 本次往返耗时（response 侧）
    retry_attempt   INTEGER NOT NULL DEFAULT 0,    -- 第几次重试（0=首次）
    error           TEXT    NOT NULL DEFAULT ''    -- 错误信息（失败时）
);
CREATE INDEX IF NOT EXISTS idx_traffic_ts ON llm_traffic(ts);
CREATE INDEX IF NOT EXISTS idx_traffic_model ON llm_traffic(model);
CREATE INDEX IF NOT EXISTS idx_traffic_ts_model ON llm_traffic(ts, model);
"""


class TrafficLogger:
    """LLM 流量存档与查询（进程内单例，SQLite 持久化）。

    线程安全：写入与查询都加锁（FastAPI 多协程/多线程访问）。
    任何异常都不影响调用方主流程（log 内部 try/except 兜底）。
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or _db_path()
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate_schema()
            self._conn.commit()
        except Exception:
            # 极端情况下（磁盘权限/路径不可写）仍可启动，流量统计静默降级
            self._conn = None

    def _migrate_schema(self) -> None:
        """彻底重新记账：旧库（缺少缓存/思考新列）直接重建空表。

        旧表只有 input_tokens / output_tokens，从未记录「缓存命中数」与「思考链」，无法精确
        迁移到 命中/未命中/输出/思考链 的新口径，因此**丢弃旧记录**、重建空表，让后续
        流量全部按新格式写入，费用估算因此精确。新库（已含新列）不做任何改动。
        """
        if self._conn is None:
            return
        try:
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(llm_traffic)").fetchall()}
            need = {"cache_hit_tokens", "cache_miss_tokens", "reasoning_tokens"}
            if not need.issubset(cols):
                # 旧库：删表重建（旧记录从未记录命中/思考链，迁移无法精确，直接清空）
                self._conn.execute("DROP TABLE IF EXISTS llm_traffic")
                self._conn.executescript(_SCHEMA)
        except Exception:
            pass

    # ---- 写入 ----

    def log(self, **attrs: Any) -> None:
        """写入一条流量记录。任何异常不影响调用方。

        常用字段见 _SCHEMA；不传的字段取默认值。provider=local 时跳过（保险过滤）。
        """
        try:
            if str(attrs.get("provider") or "") == "local":
                return
            conn = self._conn
            if conn is None:
                return
            keys = [
                "ts", "direction", "model", "provider", "api_format", "message_type",
                "session_id", "stdin_id", "cwd", "role", "thinking_level",
                "stream_supported", "bytes", "chars", "input_tokens",
                "cache_hit_tokens", "cache_miss_tokens", "output_tokens", "reasoning_tokens",
                "status_code", "duration_ms", "retry_attempt", "error",
            ]
            row = []
            for k in keys:
                v = attrs.get(k)
                if k in ("bytes", "chars", "input_tokens", "output_tokens",
                         "cache_hit_tokens", "cache_miss_tokens", "reasoning_tokens",
                         "status_code", "duration_ms", "retry_attempt", "stream_supported"):
                    row.append(int(v) if v else 0)
                elif k == "ts":
                    row.append(int(v) if v else int(time.time() * 1000))
                else:
                    row.append(str(v or ""))
            with self._lock:
                conn.execute(
                    "INSERT INTO llm_traffic (" + ",".join(keys) + ") VALUES (" + ",".join("?" * len(keys)) + ")",
                    row,
                )
                conn.commit()
        except Exception:
            pass

    def log_request(
        self,
        model: str,
        provider: str = "",
        api_format: str = "",
        message_type: str = "",
        session_id: str = "",
        stdin_id: str = "",
        cwd: str = "",
        role: str = "",
        thinking_level: str = "",
        stream_supported: bool = False,
        payload: Any = None,
        status_code: int = 0,
        retry_attempt: int = 0,
        error: str = "",
        **extra: Any,
    ) -> None:
        """写入一条 request 记录。payload 为序列化前的 Python 对象，
        内部转 JSON 计算字节数（UTF-8）与字符数。"""
        text = ""
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str) if payload is not None else ""
        except Exception:
            try:
                text = str(payload)
            except Exception:
                text = ""
        self.log(
            direction="request",
            model=model,
            provider=provider,
            api_format=api_format,
            message_type=message_type,
            session_id=session_id,
            stdin_id=stdin_id,
            cwd=cwd,
            role=role,
            thinking_level=thinking_level,
            stream_supported=1 if stream_supported else 0,
            bytes=len(text.encode("utf-8")),
            chars=len(text),
            status_code=status_code,
            retry_attempt=retry_attempt,
            error=error,
        )

    def log_response(
        self,
        model: str,
        text: str = "",
        input_tokens: int = 0,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        provider: str = "",
        api_format: str = "",
        message_type: str = "",
        session_id: str = "",
        stdin_id: str = "",
        cwd: str = "",
        role: str = "",
        thinking_level: str = "",
        stream_supported: bool = False,
        status_code: int = 200,
        duration_ms: int = 0,
        error: str = "",
        **extra: Any,
    ) -> None:
        """写入一条 response 记录。text 为模型输出正文（或非流式完整响应 JSON 文本）。"""
        self.log(
            direction="response",
            model=model,
            provider=provider,
            api_format=api_format,
            message_type=message_type,
            session_id=session_id,
            stdin_id=stdin_id,
            cwd=cwd,
            role=role,
            thinking_level=thinking_level,
            stream_supported=1 if stream_supported else 0,
            bytes=len(str(text or "").encode("utf-8")),
            chars=len(str(text or "")),
            input_tokens=input_tokens,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            status_code=status_code,
            duration_ms=duration_ms,
            error=error,
        )

    # ---- 查询 ----

    def models(self) -> List[str]:
        """出现过流量的模型列表（按最近使用时间倒序；排除 local）。"""
        try:
            with self._lock:
                if self._conn is None:
                    return []
                rows = self._conn.execute(
                    "SELECT model, MAX(ts) AS last FROM llm_traffic "
                    "WHERE provider != 'local' "
                    "GROUP BY model ORDER BY last DESC"
                ).fetchall()
                return [r[0] for r in rows]
        except Exception:
            return []

    def _bucket_expr(self, granularity: str) -> str:
        """SQLite 桶表达式（本地时区）。

        hour: 'YYYY-MM-DD HH:00'；
        day:  'YYYY-MM-DD'；
        week: 'YYYY-Wnn'（%W 周数，足够柱状图分组）。
        """
        if granularity == "day":
            return "strftime('%Y-%m-%d', ts/1000.0, 'unixepoch', 'localtime')"
        if granularity == "week":
            return "strftime('%Y', ts/1000.0, 'unixepoch', 'localtime') || '-W' || printf('%02d', CAST(strftime('%W', ts/1000.0, 'unixepoch', 'localtime') AS INTEGER))"
        return "strftime('%Y-%m-%d %H:00', ts/1000.0, 'unixepoch', 'localtime')"

    def _bucket_start(self, granularity: str, n: int) -> int:
        """最近 n 个桶的起始时间（epoch 毫秒），用于 WHERE ts >= start。

        hour：当前小时往前 n-1 小时；day：今天往前 n-1 天；week：本周往前 n-1 周。
        """
        now = datetime.now()
        if granularity == "day":
            start = datetime(now.year, now.month, now.day) - timedelta(days=n - 1)
        elif granularity == "week":
            start = datetime(now.year, now.month, now.day) - timedelta(days=now.weekday()) - timedelta(weeks=n - 1)
        else:
            start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=n - 1)
        return int(start.timestamp() * 1000)

    def _bucket_labels(self, granularity: str, n: int, start_ms: int) -> List[str]:
        """生成连续桶标签序列（与 _bucket_expr 的格式一致）。"""
        fmt = "%Y-%m-%d %H:00" if granularity == "hour" else "%Y-%m-%d"
        if granularity == "week":
            out = []
            cur = datetime.fromtimestamp(start_ms / 1000.0)
            for _ in range(n):
                out.append(cur.strftime("%Y") + "-W" + ("%02d" % int(cur.strftime("%W"))))
                cur += timedelta(weeks=1)
            return out
        out = []
        cur = datetime.fromtimestamp(start_ms / 1000.0)
        step = timedelta(hours=1) if granularity == "hour" else timedelta(days=1)
        for _ in range(n):
            out.append(cur.strftime(fmt))
            cur += step
        return out

    def stats(
        self,
        model: str = "",
        granularity: str = "hour",
        n: int = 24,
    ) -> Dict[str, Any]:
        """柱状图聚合数据。

        返回：
        {
          "granularity", "n", "model", "models": [去重模型列表，model 为空时用于堆叠],
          "buckets": [
            {"bucket": 标签, "count", "bytes_in", "bytes_out",
             "tokens_in", "tokens_in_hit", "tokens_in_miss", "tokens_out", "duration_ms",
             "by_model": {模型名: {count, bytes_in, bytes_out, tokens_in_hit, tokens_in_miss, tokens_out}}}  # 仅全部时
          ]
        }
        """
        n = max(1, min(int(n) if str(n).isdigit() else 24, 365))
        if granularity not in ("hour", "day", "week"):
            granularity = "hour"
        start_ms = self._bucket_start(granularity, n)
        labels = self._bucket_labels(granularity, n, start_ms)
        bucket_expr = self._bucket_expr(granularity)
        try:
            with self._lock:
                if self._conn is None:
                    return self._empty_stats(granularity, n, model, labels)
                where = "ts >= ? AND provider != 'local'"
                args: List[Any] = [start_ms]
                if model:
                    where += " AND model = ?"
                    args.append(model)
                if model:
                    sql = (
                        f"SELECT {bucket_expr} AS b, "
                        "COUNT(*) AS cnt, "
                        "SUM(CASE WHEN direction='request' THEN bytes ELSE 0 END) AS bin, "
                        "SUM(CASE WHEN direction='response' THEN bytes ELSE 0 END) AS bout, "
                        "SUM(CASE WHEN direction='response' THEN cache_hit_tokens ELSE 0 END) AS thin, "
                        "SUM(CASE WHEN direction='response' THEN cache_miss_tokens ELSE 0 END) AS tmin, "
                        "SUM(CASE WHEN direction='response' THEN output_tokens ELSE 0 END) AS tout, "
                        "SUM(CASE WHEN direction='response' THEN duration_ms ELSE 0 END) AS dur "
                        f"FROM llm_traffic WHERE {where} GROUP BY b ORDER BY b"
                    )
                    rows = self._conn.execute(sql, args).fetchall()
                    bucket_map: Dict[str, Dict[str, Any]] = {}
                    for r in rows:
                        hit = int(r[4] or 0)
                        miss = int(r[5] or 0)
                        bucket_map[r[0]] = {
                            "count": r[1], "bytes_in": r[2], "bytes_out": r[3],
                            "tokens_in": hit + miss, "tokens_in_hit": hit, "tokens_in_miss": miss,
                            "tokens_out": r[6], "duration_ms": r[7],
                        }
                    buckets = []
                    for lab in labels:
                        b = bucket_map.get(lab)
                        buckets.append(b or {
                            "count": 0, "bytes_in": 0, "bytes_out": 0,
                            "tokens_in": 0, "tokens_in_hit": 0, "tokens_in_miss": 0,
                            "tokens_out": 0, "duration_ms": 0,
                        })
                    return {
                        "granularity": granularity,
                        "n": n,
                        "model": model,
                        "models": [],
                        "buckets": buckets,
                    }
                # 全部模型：按 (桶, 模型) 分组，前端按模型堆叠
                sql = (
                    f"SELECT {bucket_expr} AS b, model AS m, "
                    "COUNT(*) AS cnt, "
                    "SUM(CASE WHEN direction='request' THEN bytes ELSE 0 END) AS bin, "
                    "SUM(CASE WHEN direction='response' THEN bytes ELSE 0 END) AS bout, "
                    "SUM(CASE WHEN direction='response' THEN cache_hit_tokens ELSE 0 END) AS thin, "
                    "SUM(CASE WHEN direction='response' THEN cache_miss_tokens ELSE 0 END) AS tmin, "
                    "SUM(CASE WHEN direction='response' THEN output_tokens ELSE 0 END) AS tout "
                    f"FROM llm_traffic WHERE {where} GROUP BY b, m ORDER BY b"
                )
                rows = self._conn.execute(sql, args).fetchall()
                model_order: List[str] = []
                seen: set = set()
                per_bucket: Dict[str, Dict[str, Any]] = {}
                for r in rows:
                    lab, m = r[0], r[1]
                    if m not in seen:
                        seen.add(m)
                        model_order.append(m)
                    entry = per_bucket.setdefault(lab, {
                        "count": 0, "bytes_in": 0, "bytes_out": 0,
                        "tokens_in": 0, "tokens_in_hit": 0, "tokens_in_miss": 0,
                        "tokens_out": 0, "duration_ms": 0,
                        "by_model": {},
                    })
                    entry["count"] += r[2]
                    entry["bytes_in"] += r[3]
                    entry["bytes_out"] += r[4]
                    hit = int(r[5] or 0)
                    miss = int(r[6] or 0)
                    entry["tokens_in_hit"] += hit
                    entry["tokens_in_miss"] += miss
                    entry["tokens_in"] += hit + miss
                    entry["tokens_out"] += r[7]
                    entry["by_model"][m] = {
                        "count": r[2], "bytes_in": r[3], "bytes_out": r[4],
                        "tokens_in_hit": r[5], "tokens_in_miss": r[6], "tokens_out": r[7],
                    }
                buckets = []
                for lab in labels:
                    entry = per_bucket.get(lab)
                    buckets.append(entry or {
                        "count": 0, "bytes_in": 0, "bytes_out": 0,
                        "tokens_in": 0, "tokens_in_hit": 0, "tokens_in_miss": 0,
                        "tokens_out": 0, "duration_ms": 0,
                        "by_model": {},
                    })
                return {
                    "granularity": granularity,
                    "n": n,
                    "model": "",
                    "models": model_order,
                    "buckets": buckets,
                }
        except Exception:
            return self._empty_stats(granularity, n, model, labels)

    @staticmethod
    def _empty_stats(granularity: str, n: int, model: str, labels: List[str]) -> Dict[str, Any]:
        return {
            "granularity": granularity,
            "n": n,
            "model": model,
            "models": [],
            "buckets": [
                {"count": 0, "bytes_in": 0, "bytes_out": 0,
                 "tokens_in": 0, "tokens_in_hit": 0, "tokens_in_miss": 0,
                 "tokens_out": 0, "duration_ms": 0, "by_model": {}}
                for _ in labels
            ],
        }

    def summary(
        self,
        model: str = "",
        granularity: str = "hour",
        n: int = 24,
    ) -> Dict[str, Any]:
        """汇总卡片：总次数/入出字符数/命中与未命中 tokens/出 tokens/平均耗时/失败数。

        上行流量 = 请求载荷字符数（chars，direction=request）；
        下行流量 = 模型输出字符数（chars，direction=response）。
        兼容保留 bytes_in/bytes_out（字节数），前端可切换口径。
        """
        n = max(1, min(int(n) if str(n).isdigit() else 24, 365))
        if granularity not in ("hour", "day", "week"):
            granularity = "hour"
        start_ms = self._bucket_start(granularity, n)
        try:
            with self._lock:
                if self._conn is None:
                    return self._empty_summary(start_ms, model)
                where = "ts >= ? AND provider != 'local'"
                args: List[Any] = [start_ms]
                if model:
                    where += " AND model = ?"
                    args.append(model)
                row = self._conn.execute(
                    "SELECT "
                    "COUNT(*) AS total, "
                    "SUM(CASE WHEN direction='request' THEN chars ELSE 0 END) AS cin, "
                    "SUM(CASE WHEN direction='response' THEN chars ELSE 0 END) AS cout, "
                    "SUM(CASE WHEN direction='response' THEN input_tokens ELSE 0 END) AS tin, "
                    "SUM(CASE WHEN direction='response' THEN cache_hit_tokens ELSE 0 END) AS thin, "
                    "SUM(CASE WHEN direction='response' THEN cache_miss_tokens ELSE 0 END) AS tmin, "
                    "SUM(CASE WHEN direction='response' THEN output_tokens ELSE 0 END) AS tout, "
                    "SUM(CASE WHEN direction='response' THEN duration_ms ELSE 0 END) AS dur, "
                    "SUM(CASE WHEN direction='response' THEN 1 ELSE 0 END) AS resp_cnt, "
                    "SUM(CASE WHEN status_code IN (0, 200) THEN 0 ELSE 1 END) AS fail_cnt "
                    f"FROM llm_traffic WHERE {where}",
                    args,
                ).fetchone()
                total, cin, cout, tin, thin, tmin, tout, dur, resp_cnt, fail_cnt = row
                avg_ms = int(dur / resp_cnt) if resp_cnt else 0
                return {
                    "granularity": granularity,
                    "n": n,
                    "model": model,
                    "requests": int(total),
                    "response_count": int(resp_cnt),
                    "chars_in": int(cin or 0),
                    "chars_out": int(cout or 0),
                    "bytes_in": int(cin or 0),
                    "bytes_out": int(cout or 0),
                    "tokens_in": int(tin or 0),
                    "tokens_in_hit": int(thin or 0),
                    "tokens_in_miss": int(tmin or 0),
                    "tokens_out": int(tout or 0),
                    "avg_duration_ms": avg_ms,
                    "error_count": int(fail_cnt or 0),
                    "start_ms": start_ms,
                }
        except Exception:
            return self._empty_summary(start_ms, model)

    @staticmethod
    def _empty_summary(start_ms: int, model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "requests": 0, "response_count": 0,
            "chars_in": 0, "chars_out": 0,
            "bytes_in": 0, "bytes_out": 0,
            "tokens_in": 0, "tokens_in_hit": 0, "tokens_in_miss": 0, "tokens_out": 0,
            "avg_duration_ms": 0, "error_count": 0, "start_ms": start_ms,
        }

    def per_model(
        self,
        granularity: str = "hour",
        n: int = 24,
        model: str = "",
    ) -> List[Dict[str, Any]]:
        """按模型汇总（时间范围内），供费用估算按模型单价计算。

        model 非空时只返回该模型的汇总（费用估算单模型筛选用）。
        返回 [{model, requests, chars_in, chars_out, tokens_in, tokens_out}]
        """
        n = max(1, min(int(n) if str(n).isdigit() else 24, 365))
        if granularity not in ("hour", "day", "week"):
            granularity = "hour"
        start_ms = self._bucket_start(granularity, n)
        where = "ts >= ? AND provider != 'local'"
        args: List[Any] = [start_ms]
        if model:
            where += " AND model = ?"
            args.append(model)
        try:
            with self._lock:
                if self._conn is None:
                    return []
                rows = self._conn.execute(
                    "SELECT model, "
                    "COUNT(*) AS total, "
                    "SUM(CASE WHEN direction='request' THEN chars ELSE 0 END) AS cin, "
                    "SUM(CASE WHEN direction='response' THEN chars ELSE 0 END) AS cout, "
                    "SUM(CASE WHEN direction='response' THEN cache_hit_tokens ELSE 0 END) AS thin, "
                    "SUM(CASE WHEN direction='response' THEN cache_miss_tokens ELSE 0 END) AS tmin, "
                    "SUM(CASE WHEN direction='response' THEN output_tokens ELSE 0 END) AS tout "
                    f"FROM llm_traffic WHERE {where} "
                    "GROUP BY model ORDER BY total DESC",
                    args,
                ).fetchall()
                return [
                    {
                        "model": r[0],
                        "requests": int(r[1]),
                        "chars_in": int(r[2] or 0),
                        "chars_out": int(r[3] or 0),
                        "tokens_in_hit": int(r[4] or 0),
                        "tokens_in_miss": int(r[5] or 0),
                        "tokens_in": int(r[4] or 0) + int(r[5] or 0),
                        "tokens_out": int(r[6] or 0),
                    }
                    for r in rows
                ]
        except Exception:
            return []

    def session_per_model(self, session_id: str) -> List[Dict[str, Any]]:
        """按模型汇总指定会话的流量（费用估算用，与 per_model 同口径）。

        返回 [{model, requests, chars_in, chars_out, tokens_in, tokens_out}]；
        免费/未配置单价的模型由调用方按 0 计。会话无记录返回 []。

        供「本会话费用估算」使用：/status 的 session_stats 是全模型汇总，
        这里按模型拆开，才能套用与流量统计页一致的「按模型单价估算」算法。
        """
        if not session_id:
            return []
        try:
            with self._lock:
                if self._conn is None:
                    return []
                rows = self._conn.execute(
                    "SELECT model, "
                    "COUNT(*) AS total, "
                    "SUM(CASE WHEN direction='request' THEN chars ELSE 0 END) AS cin, "
                    "SUM(CASE WHEN direction='response' THEN chars ELSE 0 END) AS cout, "
                    "SUM(CASE WHEN direction='response' THEN cache_hit_tokens ELSE 0 END) AS thin, "
                    "SUM(CASE WHEN direction='response' THEN cache_miss_tokens ELSE 0 END) AS tmin, "
                    "SUM(CASE WHEN direction='response' THEN output_tokens ELSE 0 END) AS tout "
                    "FROM llm_traffic WHERE session_id = ? AND provider != 'local' "
                    "GROUP BY model ORDER BY total DESC",
                    (session_id,),
                ).fetchall()
                return [
                    {
                        "model": r[0],
                        "requests": int(r[1]),
                        "chars_in": int(r[2] or 0),
                        "chars_out": int(r[3] or 0),
                        "tokens_in_hit": int(r[4] or 0),
                        "tokens_in_miss": int(r[5] or 0),
                        "tokens_in": int(r[4] or 0) + int(r[5] or 0),
                        "tokens_out": int(r[6] or 0),
                    }
                    for r in rows
                ]
        except Exception:
            return []

    def records(
        self,
        model: str = "",
        direction: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页明细（最近在前）。model/direction 可空 = 不筛选。"""
        page = max(1, int(page) if str(page).isdigit() else 1)
        page_size = min(max(1, int(page_size) if str(page_size).isdigit() else 20), 100)
        try:
            with self._lock:
                if self._conn is None:
                    return {"total": 0, "page": page, "page_size": page_size, "records": []}
                where = "provider != 'local'"
                args: List[Any] = []
                if model:
                    where += " AND model = ?"
                    args.append(model)
                if direction in ("request", "response"):
                    where += " AND direction = ?"
                    args.append(direction)
                total = self._conn.execute(
                    f"SELECT COUNT(*) FROM llm_traffic WHERE {where}", args
                ).fetchone()[0]
                offset = (page - 1) * page_size
                rows = self._conn.execute(
                    f"SELECT ts, direction, model, provider, api_format, message_type, "
                    "session_id, stdin_id, cwd, role, thinking_level, stream_supported, "
                    "bytes, chars, input_tokens, cache_hit_tokens, cache_miss_tokens, "
                    "output_tokens, reasoning_tokens, status_code, duration_ms, "
                    "retry_attempt, error "
                    f"FROM llm_traffic WHERE {where} ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
                    args + [page_size, offset],
                ).fetchall()
                keys = [
                    "ts", "direction", "model", "provider", "api_format", "message_type",
                    "session_id", "stdin_id", "cwd", "role", "thinking_level",
                    "stream_supported", "bytes", "chars", "input_tokens",
                    "cache_hit_tokens", "cache_miss_tokens", "output_tokens", "reasoning_tokens",
                    "status_code", "duration_ms", "retry_attempt", "error",
                ]
                records = [dict(zip(keys, r)) for r in rows]
                return {"total": int(total), "page": page, "page_size": page_size, "records": records}
        except Exception:
            return {"total": 0, "page": page, "page_size": page_size, "records": []}

    def last_request_chars(self, session_id: str) -> Optional[int]:
        """返回指定会话最近一条已发出的 request 的字符数（chars）。

        用于「当前上下文占用」显示：取本会话最近一次真实发出的请求体字符数，
        按 1 字符 ≈ 0.6 token 折算即可作为上下文 token 估算（与流量库同口径）。

        Args:
            session_id: 会话 id（如 desk_xxx / session_xxx，与 llm_traffic.session_id 一致）。

        Returns:
            最近一条 request 的 chars；本会话尚无 request 记录时返回 None。
        """
        if not session_id:
            return None
        try:
            with self._lock:
                if self._conn is None:
                    return None
                row = self._conn.execute(
                    "SELECT chars FROM llm_traffic "
                    "WHERE session_id = ? AND direction = 'request' AND provider != 'local' "
                    "ORDER BY ts DESC, id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                return int(row[0]) if row else None
        except Exception:
            return None

    def session_stats(self, session_id: str) -> Dict[str, Any]:
        """指定会话的流量摘要（供 /status 查看，与是否活跃无关）。

        chars_in   = 本会话所有请求载荷的字符数之和（估算输入 tokens）；
        tokens_out = 本会话所有响应的 output_tokens 之和（实际输出 tokens）；
        tokens_in  = 本会话所有响应的 input_tokens 之和（网关参考值）；
        cache_hit_tokens = 本会话所有响应命中上下文缓存的输入 token 数之和；
        requests / responses = 请求/响应条数；duration_ms = 累计耗时。
        """
        empty: Dict[str, Any] = {
            "session_id": session_id, "chars_in": 0, "tokens_in": 0, "tokens_out": 0,
            "cache_hit_tokens": 0, "requests": 0, "responses": 0, "duration_ms": 0,
        }
        if not session_id:
            return empty
        try:
            with self._lock:
                if self._conn is None:
                    return empty
                row = self._conn.execute(
                    "SELECT "
                    "SUM(CASE WHEN direction='request' THEN chars ELSE 0 END) AS chars_in, "
                    "SUM(CASE WHEN direction='response' THEN input_tokens ELSE 0 END) AS tokens_in, "
                    "SUM(CASE WHEN direction='response' THEN output_tokens ELSE 0 END) AS tokens_out, "
                    "SUM(CASE WHEN direction='response' THEN cache_hit_tokens ELSE 0 END) AS cache_hit, "
                    "SUM(CASE WHEN direction='request' THEN 1 ELSE 0 END) AS req_cnt, "
                    "SUM(CASE WHEN direction='response' THEN 1 ELSE 0 END) AS resp_cnt, "
                    "SUM(CASE WHEN direction='response' THEN duration_ms ELSE 0 END) AS dur "
                    "FROM llm_traffic WHERE session_id = ? AND provider != 'local'",
                    (session_id,),
                ).fetchone()
                if not row:
                    return empty
                return {
                    "session_id": session_id,
                    "chars_in": int(row[0] or 0),
                    "tokens_in": int(row[1] or 0),
                    "tokens_out": int(row[2] or 0),
                    "cache_hit_tokens": int(row[3] or 0),
                    "requests": int(row[4] or 0),
                    "responses": int(row[5] or 0),
                    "duration_ms": int(row[6] or 0),
                }
        except Exception:
            return empty

    def clear(self, model: str = "") -> bool:
        """清空指定模型（或全部 when model=''）。返回是否删除到数据。"""
        try:
            with self._lock:
                if self._conn is None:
                    return False
                if model:
                    cur = self._conn.execute("DELETE FROM llm_traffic WHERE model = ?", (model,))
                else:
                    cur = self._conn.execute("DELETE FROM llm_traffic")
                self._conn.commit()
                return cur.rowcount > 0
        except Exception:
            return False


# 进程内全局单例：engine/ / server.py 通过它记录与查询
TRAFFIC = TrafficLogger()
