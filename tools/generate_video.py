"""
generate_video 工具：按 models.json video_models 的 kind 分发。
对外参数统一为 OpenAI Videos 兼容风格（mode/seconds/size/aspect_ratio + 媒体），
内部按当前激活模型自适应：
- agnes-v2.5（Agnes Video 2.5 Flash）：size 固定归一为 "720P"；seconds 字符串 "4"-"12"；
  mode=keyframe 用 first_frame/last_frame，mode=reference 用 images（≤5）/audios。
- agnes-v2.0（默认，Agnes Video V2.0）：把 size/aspect_ratio 换算成 width/height、
  seconds 换算成 num_frames，媒体取 first_frame 或 images 的首张给 image。

流程：POST <api_url>（如 /v1/videos）创建任务 → 轮询 GET <query_url>?video_id=...&model_name=...
（默认由 api_url 推导 host + /agnesapi）→ status=completed 后从 metadata.url 下载保存到
<cwd>/.bigcodex_uploads/。
"""
import base64
import mimetypes
import os
import re
import time
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext
from video_config import VIDEO_MODELS, resolve_video_model

CREATE_TIMEOUT = 60        # 创建任务超时（秒）
POLL_INTERVAL = 5          # 轮询间隔（秒，agnes-v2.0）
POLL_INTERVAL_FAST = 2     # 轮询间隔（秒，agnes-v2.5，官方建议 1-2 秒）
POLL_TIMEOUT = 600         # 轮询总超时（秒）
DOWNLOAD_TIMEOUT = 180     # 下载视频超时（秒）
MAX_FRAMES = 441           # 最大帧数（agnes-v2.0）
FLASH_SIZE = "720P"        # Agnes Video 2.5 Flash 固定分辨率档位
FLASH_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
FLASH_MODES = ("text", "keyframe", "reference")
FLASH_SECONDS_MIN = 4
FLASH_SECONDS_MAX = 12
FLASH_MAX_IMAGES = 5       # reference 模式 images 最多 5 张
V20_DEFAULT_WH = {
    "480P": (832, 448),
    "720P": (1280, 720),
    "1080P": (1920, 1080),
}
V20_TIER_HEIGHT = {
    "480P": 448,
    "720P": 720,
    "1080P": 1080,
}


class GenerateVideoTool(BaseTool):
    """generate_video 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="generate_video",
            description=(
                "生成视频（异步任务，通常耗时 1-5 分钟）。参数统一为 OpenAI Videos 兼容风格，"
                "会根据当前激活的视频模型（Agnes Video V2.0 / V2.5 Flash）自动适配内部参数。\n"
                "参数：prompt（必填，视频内容描述）；mode（text/keyframe/reference，默认 text）；"
                "seconds（时长，\"4\"-\"12\"，默认 \"5\"）；"
                "size（分辨率档位，如 \"480P\"/\"720P\"/\"1080P\" 或 \"宽x高\"；默认 720P，其中 "
                "V2.5 固定 720P，V2.0 会按画幅换算成内部宽高）；"
                "aspect_ratio（画幅，21:9/16:9/4:3/1:1/3:4/9:16，默认 16:9）；"
                "keyframe 模式用 first_frame/last_frame（图片 URL 或本地路径，至少一个）；"
                "reference 模式用 images（≤5，图片 URL 或本地路径）/audios；seed、n 可选。\n"
                "生成结果保存到 .bigcodex_uploads/ 并返回本地路径。注意：模型免费额度有限，非必要不要连续生成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "视频内容的文字描述"},
                    "mode": {
                        "type": "string",
                        "enum": ["text", "keyframe", "reference"],
                        "description": "生成模式：text 纯文本、keyframe 首尾帧、reference 图片/音频参考",
                    },
                    "seconds": {"type": "string", "description": "时长（秒，字符串 \"4\"-\"12\"，默认 \"5\"）"},
                    "size": {
                        "type": "string",
                        "description": "分辨率档位（\"480P\"/\"720P\"/\"1080P\" 或 \"宽x高\"，默认 720P）",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "画幅（21:9/16:9/4:3/1:1/3:4/9:16，默认 16:9）",
                    },
                    "first_frame": {"type": "string", "description": "首帧图片 URL 或本地路径（keyframe 模式）"},
                    "last_frame": {"type": "string", "description": "尾帧图片 URL 或本地路径（keyframe 模式）"},
                    "images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参考图片 URL/本地路径列表（reference 模式，≤5 张）",
                    },
                    "audios": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参考音频 URL/本地路径列表（reference 模式）",
                    },
                    "seed": {"type": "integer", "description": "随机种子"},
                    "n": {"type": "integer", "description": "生成数量（当前仅支持 1，默认 1）"},
                },
                "required": ["prompt"],
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.DEFAULT,
            aliases={},
        )

    @classmethod
    def execute(
        cls,
        context: ToolContext,
        prompt: str,
        mode: Optional[str] = None,
        seconds: Optional[str] = None,
        size: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        first_frame: Optional[str] = None,
        last_frame: Optional[str] = None,
        images: Optional[List[str]] = None,
        audios: Optional[List[str]] = None,
        seed: Optional[int] = None,
        n: Optional[int] = None,
    ) -> str:
        """生成视频（异步任务 + 轮询），并把结果下载保存到 .bigcodex_uploads/。"""
        prompt = (prompt or "").strip()
        if not prompt:
            return "错误：prompt 不能为空"
        if not VIDEO_MODELS:
            return "错误：未配置视频生成模型（请在 models.json 的 video_models 中添加条目）"
        cfg = resolve_video_model()
        api_key = str(cfg.get("api_key") or "").strip()
        if not api_key:
            return "错误：generate_video 未配置 API key（请在 src/agent/.env 中设置 AGNES_API_KEY 或对应环境变量）"

        kind = str(cfg.get("kind") or "agnes-v2.0")
        if kind in ("agnes-v2.5", "agnes-v2.5-flash"):
            return cls._run_agn25(
                context, cfg, prompt, mode, seconds, size, aspect_ratio,
                first_frame, last_frame, images, audios, seed, n,
            )
        return cls._run_agn20(
            context, cfg, prompt, mode, seconds, size, aspect_ratio,
            first_frame, last_frame, images, audios, seed, n,
        )

    @classmethod
    def _run_agn20(
        cls,
        context: ToolContext,
        cfg: dict,
        prompt: str,
        mode: Optional[str],
        seconds: Optional[str],
        size: Optional[str],
        aspect_ratio: Optional[str],
        first_frame: Optional[str],
        last_frame: Optional[str],
        images: Optional[List[str]],
        audios: Optional[List[str]],
        seed: Optional[int],
        n: Optional[int],
    ) -> str:
        """Agnes Video V2.0 协议：把统一的 size/aspect_ratio/seconds 换算成宽高与帧数。"""
        mode_val = str(mode or "text").strip().lower()
        if mode_val not in FLASH_MODES:
            return "错误：mode 只能是 text/keyframe/reference"
        size_val, ratio_val, secs = cls._public_resolved(size, aspect_ratio, seconds, cfg)
        if not (size and str(size).strip()) and not str(cfg.get("size") or "").strip():
            size_val = cls._v20_default_tier(cfg)
        w, h = cls._resolve_v20_wh(size_val, ratio_val, cfg)
        fps = cls._clamp_int(cfg.get("frame_rate"), 1, 60, 24)
        frames = cls._frames_for_duration(secs, fps)

        payload = {
            "model": cfg.get("model") or "agnes-video-v2.0",
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_frames": frames,
            "frame_rate": fps,
        }
        # 媒体映射：2.0 只接受单个 image；keyframe 取首帧（或尾帧），reference 取首张参考图。
        src_image = ""
        if mode_val == "keyframe":
            if first_frame and str(first_frame).strip():
                src_image = str(first_frame).strip()
            elif last_frame and str(last_frame).strip():
                src_image = str(last_frame).strip()
            else:
                return "错误：keyframe 模式必须至少提供 first_frame 或 last_frame 之一"
        elif mode_val == "reference":
            imgs = cls._to_list(images)
            if imgs:
                src_image = imgs[0]
            elif cls._to_list(audios):
                return "错误：当前 2.0 模型不支持音频参考，请改用 images"
            else:
                return "错误：reference 模式必须至少提供 images 或 audios 之一非空"
        if src_image:
            prepared = cls._prepare_media(context, src_image)
            if isinstance(prepared, str) and prepared.startswith("错误："):
                return prepared
            payload["image"] = prepared
        if seed is not None:
            payload["seed"] = cls._clamp_int(seed, 0, 9999999999, None)
        return cls._create_and_wait(context, cfg, payload, POLL_INTERVAL)

    @classmethod
    def _run_agn25(
        cls,
        context: ToolContext,
        cfg: dict,
        prompt: str,
        mode: Optional[str],
        seconds: Optional[str],
        size: Optional[str],
        aspect_ratio: Optional[str],
        first_frame: Optional[str],
        last_frame: Optional[str],
        images: Optional[List[str]],
        audios: Optional[List[str]],
        seed: Optional[int],
        n: Optional[int],
    ) -> str:
        """Agnes Video 2.5 Flash 协议：OpenAI Videos 兼容，size 固定归一为 720P。"""
        mode_val = str(mode or "text").strip().lower()
        if mode_val not in FLASH_MODES:
            return "错误：mode 只能是 text/keyframe/reference"
        # 2.5 只支持 720P：任何 size（480P/1080P/宽x高等）都归一为 720P
        _size_val, ratio_val, secs = cls._public_resolved(size, aspect_ratio, seconds, cfg)
        if ratio_val not in FLASH_ASPECT_RATIOS:
            ratio_val = str(cfg.get("aspect_ratio") or "16:9")
        n_val = cls._clamp_int(n, 1, 1, 1)

        payload = {
            "model": cfg.get("model") or "agnes-video-2.5-flash",
            "prompt": prompt,
            "seconds": secs,
            "mode": mode_val,
            "size": FLASH_SIZE,
            "aspect_ratio": ratio_val,
        }
        if seed is not None:
            payload["seed"] = cls._clamp_int(seed, 0, 9999999999, None)
        if n_val is not None:
            payload["n"] = n_val

        if mode_val == "keyframe":
            ff = cls._prepare_media(context, str(first_frame).strip()) if first_frame and str(first_frame).strip() else ""
            lf = cls._prepare_media(context, str(last_frame).strip()) if last_frame and str(last_frame).strip() else ""
            if isinstance(ff, str) and ff.startswith("错误："):
                return ff
            if isinstance(lf, str) and lf.startswith("错误："):
                return lf
            if not ff and not lf:
                return "错误：keyframe 模式必须至少提供 first_frame 或 last_frame 之一"
            if ff:
                payload["first_frame"] = ff
            if lf:
                payload["last_frame"] = lf
        elif mode_val == "reference":
            imgs = cls._prepare_media_list(context, images, "reference 图片")
            if isinstance(imgs, str) and imgs.startswith("错误："):
                return imgs
            auds = cls._prepare_media_list(context, audios, "参考音频")
            if isinstance(auds, str) and auds.startswith("错误："):
                return auds
            if not imgs and not auds:
                return "错误：reference 模式必须至少提供 images 或 audios 之一非空"
            if len(imgs) > FLASH_MAX_IMAGES:
                return f"错误：images 最多支持 {FLASH_MAX_IMAGES} 张（实际 {len(imgs)} 张）"
            if imgs:
                payload["images"] = imgs
            if auds:
                payload["audios"] = auds
        else:  # text
            if first_frame or last_frame or images or audios:
                return "错误：text 模式不允许传入 first_frame/last_frame/images/audios"
        return cls._create_and_wait(context, cfg, payload, POLL_INTERVAL_FAST)

    @classmethod
    def _public_resolved(cls, size, aspect_ratio, seconds, cfg):
        """归一化统一的公开参数为 (size, aspect_ratio, seconds)，供两种 kind 翻译使用。"""
        size_val = str(size or "").strip()
        if not size_val:
            size_val = str(cfg.get("size") or "").strip() or "720P"
        ratio_val = str(aspect_ratio or "").strip() or str(cfg.get("aspect_ratio") or "").strip() or "16:9"
        secs = cls._clamp_seconds(seconds, str(cfg.get("seconds") or "5"))
        return size_val, ratio_val, secs

    @classmethod
    def _v20_default_tier(cls, cfg: dict) -> str:
        """未指定 size 时，按 2.0 配置的默认分辨率（video_height）推导档位。"""
        h = int(cfg.get("video_height") or 448)
        for tier, th in (("1080P", 1080), ("720P", 720), ("480P", 448)):
            if h >= th:
                return tier
        return "480P"

    @classmethod
    def _resolve_v20_wh(cls, size: str, aspect_ratio: str, cfg: dict):
        """把统一的 size（档位或宽x高）+ aspect_ratio 换算成 V2.0 的 width/height。"""
        size_val = (size or "").strip()
        m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", size_val, re.IGNORECASE)
        if m:
            w = cls._clamp_int(int(m.group(1)), 64, 4096, None)
            h = cls._clamp_int(int(m.group(2)), 64, 4096, None)
            if w is not None and h is not None:
                return w, h
        ar_w, ar_h = cls._parse_ar(aspect_ratio)
        tier = size_val.upper()
        if ar_w == 16 and ar_h == 9:
            return V20_DEFAULT_WH.get(
                tier,
                (int(cfg.get("video_width") or 832), int(cfg.get("video_height") or 448)),
            )
        height = V20_TIER_HEIGHT.get(tier, int(cfg.get("video_height") or 448))
        width = cls._clamp_int(round(height * ar_w / ar_h), 64, 4096, 832)
        if width is None:
            width = int(cfg.get("video_width") or 832)
        return width, height

    @classmethod
    def _parse_ar(cls, aspect_ratio: str):
        """解析画幅字符串为 (w, h)，非法时回退 16:9。"""
        try:
            ar_w, ar_h = str(aspect_ratio).lower().split(":")
            return int(ar_w), int(ar_h)
        except Exception:
            return 16, 9

    @classmethod
    def _create_and_wait(cls, context: ToolContext, cfg: dict, payload: dict, poll_interval: int) -> str:
        """创建视频任务 → 轮询查询 → 下载保存，供两种 kind 共用。"""
        api_key = str(cfg.get("api_key") or "").strip()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = str(cfg.get("api_url") or "https://apihub.agnes-ai.com/v1/videos")
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=CREATE_TIMEOUT)
        except Exception as e:
            return f"错误：创建视频任务失败：{e}"
        if resp.status_code != 200:
            return f"错误：创建视频任务返回 {resp.status_code}：{resp.text[:300]}"
        try:
            task = resp.json()
        except Exception:
            return f"错误：创建视频任务响应解析失败：{resp.text[:300]}"
        video_id = str(task.get("video_id") or task.get("task_id") or task.get("id") or "").strip()
        if not video_id:
            return f"错误：创建视频任务响应缺少 video_id：{str(task)[:300]}"

        query_url = cls._query_url(cfg, video_id)
        started = time.time()
        result = None
        while time.time() - started < POLL_TIMEOUT:
            time.sleep(poll_interval)
            try:
                qr = httpx.get(query_url, headers=headers, timeout=CREATE_TIMEOUT)
                if qr.status_code != 200:
                    continue
                result = qr.json()
            except Exception:
                continue
            status = str((result or {}).get("status") or "")
            if status == "completed":
                break
            if status == "failed":
                return f"错误：视频生成失败：{result.get('error') or result}"

        if not result or str(result.get("status") or "") != "completed":
            return "错误：视频生成超时（任务未在限时内完成），可稍后重试"

        meta = result.get("metadata") or {}
        # 实际接口返回顶层 url 字段（文档示例为 metadata.url，兼容两者）
        video_url = str(meta.get("url") or result.get("url") or "").strip()
        if not video_url:
            return f"错误：视频任务完成但缺少 metadata.url：{str(result)[:300]}"

        out_dir = os.path.join(context.cwd, ".bigcodex_uploads")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            return f"错误：创建保存目录失败：{e}"
        try:
            r = httpx.get(video_url, timeout=DOWNLOAD_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            return f"错误：下载视频失败：{e}"
        ext = cls._guess_ext(video_url, r.headers.get("content-type", ""))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"video_{stamp}{ext}")
        try:
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            return f"错误：保存视频失败：{e}"
        return f"已生成视频，保存在 .bigcodex_uploads/：\n- {path}"

    @classmethod
    def _prepare_media(cls, context: ToolContext, value: str) -> str:
        """把输入媒体（图片/音频）转换为接口可用的值：http(s) 链接原样透传，本地文件转 base64 data URL。"""
        if not value:
            return ""
        if value.lower().startswith(("http://", "https://")):
            return value
        try:
            resolved = context.resolve_path(value, "generate_video")
        except Exception as e:
            return f"错误：{e}"
        if not os.path.isfile(resolved):
            return f"错误：媒体文件不存在（{resolved}）"
        mime = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
        try:
            with open(resolved, "rb") as f:
                raw = f.read()
        except Exception as e:
            return f"错误：读取媒体文件失败（{resolved}）：{e}"
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    @classmethod
    def _prepare_media_list(cls, context: ToolContext, items, label: str):
        """把媒体列表逐个转换为接口可用的值；返回列表，任一失败返回错误字符串。"""
        items = cls._to_list(items)
        if not items:
            return []
        out = []
        for item in items:
            prepared = cls._prepare_media(context, item)
            if isinstance(prepared, str) and prepared.startswith("错误："):
                return prepared
            out.append(prepared)
        return out

    @classmethod
    def _to_list(cls, value) -> List[str]:
        """把值归一化为字符串列表：列表/元组原样，字符串按逗号/换行拆分。"""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [v.strip() for v in value.replace("\n", ",").split(",") if v.strip()]
        return [str(value)]

    @classmethod
    def _clamp_seconds(cls, seconds, default: str) -> str:
        """把用户指定的时长（秒）钳制到 [4, 12]，以字符串返回（Flash 要求字符串）。"""
        try:
            s = int(float(str(seconds).strip()))
        except (TypeError, ValueError):
            try:
                s = int(float(str(default).strip() or "5"))
            except (TypeError, ValueError):
                s = 5
        s = max(FLASH_SECONDS_MIN, min(FLASH_SECONDS_MAX, s))
        return str(s)

    @classmethod
    def _query_url(cls, cfg: dict, video_id: str) -> str:
        """构造任务查询 URL：配置 query_url 优先，否则由 api_url 推导 host + /agnesapi。"""
        q = str(cfg.get("query_url") or "").strip()
        if not q:
            api = str(cfg.get("api_url") or "").rstrip("/")
            parsed = urlparse(api)
            q = f"{parsed.scheme}://{parsed.netloc}/agnesapi"
        sep = "&" if "?" in q else "?"
        return f"{q}{sep}video_id={video_id}&model_name={cfg.get('model') or ''}"

    @classmethod
    def _clamp_frames(cls, value, default: int) -> int:
        """帧数钳制到 [9, 441] 并规整为 8n+1。"""
        try:
            f = int(value)
        except (TypeError, ValueError):
            f = default
        f = max(9, min(MAX_FRAMES, f))
        return ((f - 1) // 8) * 8 + 1

    @classmethod
    def _frames_for_duration(cls, seconds, frame_rate: int) -> int:
        """把用户指定的时长（秒）对齐到支持档位：约 3/5/10/18 秒（81/121/241/441 帧，
        按目标帧数向上取整到下一档；超出最大档位时取 441 帧 ≈ 18 秒）。"""
        tiers = (81, 121, 241, MAX_FRAMES)
        target = float(seconds) * frame_rate
        for f in tiers:
            if f >= target:
                return f
        return MAX_FRAMES

    @classmethod
    def _guess_ext(cls, video_url: str, content_type: str) -> str:
        """根据 URL 后缀或 Content-Type 推断视频扩展名，默认 .mp4。"""
        path = urlparse(video_url).path or ""
        suffix = os.path.splitext(path)[1].lower()
        if suffix in (".mp4", ".webm", ".mov", ".mkv", ".avi"):
            return suffix
        ctype = (content_type or "").split(";")[0].strip().lower()
        mapping = {
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            "video/quicktime": ".mov",
            "video/x-matroska": ".mkv",
        }
        return mapping.get(ctype, ".mp4")

    @classmethod
    def _clamp_int(cls, value, lo: int, hi: int, default):
        try:
            return max(lo, min(hi, int(value)))
        except (TypeError, ValueError):
            return default
