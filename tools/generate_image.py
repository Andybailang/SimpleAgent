"""
generate_image 工具：按 models.json image_models 的 kind 分发：
- kolors（SiliconFlow Kwai-Kolors/Kolors）：image_size 精确分辨率、images[] 响应；
- agnes-image（Agnes Image 2.1 Flash）：size 档位 + ratio、extra_body 参数、
  data[].url / b64_json 响应（默认 url 输出）。
文生图：给出 prompt；图生图：额外传入 image（本地路径或 http(s) 链接，
本地文件转 data:image/...;base64）。
生成结果立即下载保存到 <cwd>/.bigcodex_uploads/（URL 有效期约 1 小时）。
"""
import base64
import mimetypes
import os
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx

from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext
from image_config import IMAGE_MODELS, resolve_image_model

# Kolors 官方支持的分辨率（widthxheight）
KOLORS_ALLOWED_SIZES = ("1024x1024", "960x1280", "768x1024", "720x1440", "720x1280")
AGNES_IMAGE_TIERS = ("1k", "2k", "3k", "4k")

GENERATE_TIMEOUT = 180  # 生成接口超时（秒）
DOWNLOAD_TIMEOUT = 120  # 下载结果图片超时（秒）


class GenerateImageTool(BaseTool):
    """generate_image 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="generate_image",
            description=(
                "使用当前激活的图片生成模型（Kolors / Agnes Image 2.1 Flash）生成或处理图片，支持中英文提示词。"
                "参数：prompt（必填，文字描述）；image（可选，本地图片路径或 http(s) 链接，"
                "用于图生图/图片处理）；image_size（可选：Kolors 用 1024x1024/960x1280/768x1024/720x1440/720x1280，"
                "Agnes 用 1K/2K/3K/4K 档位或 宽x高）；ratio（可选，Agnes 宽高比，如 16:9，默认 1:1）；"
                "count（可选 1-4，默认 1，仅 Kolors 生效）；"
                "num_inference_steps（可选 1-100，默认 20）；guidance_scale（可选 0-20，默认 7.5）；"
                "negative_prompt、seed 可选。生成结果保存到 .bigcodex_uploads/ 并返回本地路径。"
                "注意：模型免费额度有限，非必要不要连续生成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "生成图片的文字描述（支持中英文）"},
                    "image": {"type": "string", "description": "可选：本地图片路径或 http(s) 链接，用于图生图/图片处理"},
                    "image_size": {"type": "string", "description": "尺寸：Kolors 用 1024x1024 等精确分辨率；Agnes 用 1K/2K/3K/4K 档位或 宽x高"},
                    "ratio": {"type": "string", "description": "可选：Agnes 宽高比（1:1/16:9/9:16/4:3/3:4/2:3/3:2/21:9，默认 1:1）"},
                    "count": {"type": "integer", "description": "生成数量（1-4，默认 1）"},
                    "num_inference_steps": {"type": "integer", "description": "推理步数（1-100，默认 20）"},
                    "guidance_scale": {"type": "number", "description": "提示词引导强度（0-20，默认 7.5），越高越贴合 prompt"},
                    "negative_prompt": {"type": "string", "description": "可选：不希望出现在图片中的内容"},
                    "seed": {"type": "integer", "description": "可选：随机种子（0-9999999999），固定种子可复现结果"},
                },
                "required": ["prompt"],
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.DEFAULT,
            aliases={"image": ["image_path", "input_image"]},
        )

    @classmethod
    def execute(
        cls,
        context: ToolContext,
        prompt: str,
        image: Optional[str] = None,
        image_size: Optional[str] = None,
        count: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        ratio: Optional[str] = None,
    ) -> str:
        """生成或处理图片，并把结果下载保存到 .bigcodex_uploads/。"""
        prompt = (prompt or "").strip()
        if not prompt:
            return "错误：prompt 不能为空"
        if not IMAGE_MODELS:
            return ("错误：未配置图片生成模型（请在 models.json 的 image_models 中添加条目，"
                    "例如 SiliconFlow 的 Kwai-Kolors/Kolors 或 Agnes Image 2.1 Flash）")
        cfg = resolve_image_model()
        if (cfg.get("kind") or "kolors") == "agnes-image":
            return cls._run_agnes(
                context, cfg, prompt, image, image_size, ratio,
                num_inference_steps, negative_prompt, seed,
            )
        return cls._run_kolors(
            context, cfg, prompt, image, image_size, count,
            num_inference_steps, guidance_scale, negative_prompt, seed,
        )

    @classmethod
    def _run_kolors(
        cls,
        context: ToolContext,
        cfg: dict,
        prompt: str,
        image: Optional[str],
        image_size: Optional[str],
        count: Optional[int],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        negative_prompt: Optional[str],
        seed: Optional[int],
    ) -> str:
        """Kolors 协议：精确分辨率、image_size/batch_size/guidance_scale、images[] 响应。"""
        api_key = str(cfg.get("api_key") or "").strip()
        if not api_key:
            return ("错误：generate_image 未配置 API key（请在 src/agent/.env 中设置 "
                    "SILICONFLOW_API_KEY，或设置环境变量 SEMANTIC_SEARCH_API_KEY）")

        size = (image_size or cfg.get("image_size") or "1024x1024").strip().lower()
        if size not in KOLORS_ALLOWED_SIZES:
            return f"错误：不支持的 image_size {size}，可选：{', '.join(KOLORS_ALLOWED_SIZES)}"
        n = cls._clamp_int(count, 1, 4, int(cfg.get("batch_size") or 1))
        steps = cls._clamp_int(num_inference_steps, 1, 100, int(cfg.get("num_inference_steps") or 20))
        guidance = cls._clamp_float(guidance_scale, 0.0, 20.0, float(cfg.get("guidance_scale") or 7.5))

        payload = {
            "model": cfg.get("model") or "Kwai-Kolors/Kolors",
            "prompt": prompt,
            "image_size": size,
            "batch_size": n,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
        if image and str(image).strip():
            prepared = cls._prepare_input_image(context, str(image).strip())
            if isinstance(prepared, str) and prepared.startswith("错误："):
                return prepared
            payload["image"] = prepared
        if negative_prompt and str(negative_prompt).strip():
            payload["negative_prompt"] = str(negative_prompt).strip()
        if seed is not None:
            payload["seed"] = cls._clamp_int(seed, 0, 9999999999, None)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = str(cfg.get("api_url") or "https://api.siliconflow.cn/v1/images/generations")
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=GENERATE_TIMEOUT)
        except Exception as e:
            return f"错误：调用图片生成 API 失败：{e}"
        if resp.status_code != 200:
            return f"错误：图片生成 API 返回 {resp.status_code}：{resp.text[:300]}"
        try:
            data = resp.json()
        except Exception:
            return f"错误：图片生成 API 响应解析失败：{resp.text[:300]}"
        images = data.get("images") or []
        if not images:
            return f"错误：图片生成 API 响应缺少 images 字段：{str(data)[:300]}"

        out_dir = os.path.join(context.cwd, ".bigcodex_uploads")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            return f"错误：创建保存目录失败：{e}"

        saved: list = []
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, item in enumerate(images, 1):
            img_url = ""
            if isinstance(item, dict):
                img_url = item.get("url") or item.get("image") or ""
            elif isinstance(item, str):
                img_url = item
            if not img_url:
                return f"错误：第 {i} 张图片缺少 url：{str(item)[:200]}"
            try:
                r = httpx.get(img_url, timeout=DOWNLOAD_TIMEOUT)
                r.raise_for_status()
            except Exception as e:
                return f"错误：下载第 {i} 张图片失败：{e}"
            ext = cls._guess_ext(img_url, r.headers.get("content-type", ""))
            path = os.path.join(out_dir, f"kolors_{stamp}_{i}{ext}")
            try:
                with open(path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                return f"错误：保存第 {i} 张图片失败：{e}"
            saved.append(path)

        title = f"已生成 {len(saved)} 张图片，保存在 .bigcodex_uploads/：" if len(saved) > 1 \
            else "已生成图片，保存在 .bigcodex_uploads/："
        return title + "\n" + "\n".join(f"- {p}" for p in saved)

    @classmethod
    def _run_agnes(
        cls,
        context: ToolContext,
        cfg: dict,
        prompt: str,
        image: Optional[str],
        image_size: Optional[str],
        ratio: Optional[str],
        num_inference_steps: Optional[int],
        negative_prompt: Optional[str],
        seed: Optional[int],
    ) -> str:
        """Agnes Image 2.1 Flash 协议：size 档位 + ratio、extra_body 参数、data[].url/b64_json 响应。"""
        api_key = str(cfg.get("api_key") or "").strip()
        if not api_key:
            return ("错误：generate_image 未配置 API key（请在 src/agent/.env 中设置 "
                    "AGNES_API_KEY 或对应环境变量）")
        size = (image_size or cfg.get("image_size") or "1K").strip()
        if size.lower() not in AGNES_IMAGE_TIERS and not re.fullmatch(r"\d+x\d+", size, re.IGNORECASE):
            return f"错误：不支持的 image_size {size}，Agnes 支持 1K/2K/3K/4K 或 宽x高"
        ratio_val = (ratio or cfg.get("ratio") or "1:1").strip()

        payload: dict = {
            "model": cfg.get("model") or "agnes-image-2.1-flash",
            "prompt": prompt,
            "size": size,
        }
        if ratio_val:
            payload["ratio"] = ratio_val
        extra: dict = {"response_format": "url"}
        if image and str(image).strip():
            prepared = cls._prepare_input_image(context, str(image).strip())
            if isinstance(prepared, str) and prepared.startswith("错误："):
                return prepared
            extra["image"] = [prepared]
        if num_inference_steps is not None:
            payload["num_inference_steps"] = cls._clamp_int(num_inference_steps, 1, 100, None)
        if negative_prompt and str(negative_prompt).strip():
            payload["negative_prompt"] = str(negative_prompt).strip()
        if seed is not None:
            payload["seed"] = cls._clamp_int(seed, 0, 9999999999, None)
        payload["extra_body"] = extra

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = str(cfg.get("api_url") or "https://apihub.agnes-ai.com/v1/images/generations")
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=GENERATE_TIMEOUT)
        except Exception as e:
            return f"错误：调用图片生成 API 失败：{e}"
        if resp.status_code != 200:
            return f"错误：图片生成 API 返回 {resp.status_code}：{resp.text[:300]}"
        try:
            data = resp.json()
        except Exception:
            return f"错误：图片生成 API 响应解析失败：{resp.text[:300]}"
        items = data.get("data") or []
        if not items:
            return f"错误：Agnes 响应缺少 data 字段：{str(data)[:300]}"

        out_dir = os.path.join(context.cwd, ".bigcodex_uploads")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            return f"错误：创建保存目录失败：{e}"

        saved: list = []
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{cfg.get('id') or 'image'}_"
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                return f"错误：第 {i} 张图片响应格式异常：{str(item)[:200]}"
            img_url = item.get("url") or ""
            b64 = item.get("b64_json") or ""
            if not img_url and not b64:
                return f"错误：第 {i} 张图片缺少 url/b64_json：{str(item)[:200]}"
            ext = ".png"
            try:
                if img_url:
                    r = httpx.get(img_url, timeout=DOWNLOAD_TIMEOUT)
                    r.raise_for_status()
                    ext = cls._guess_ext(img_url, r.headers.get("content-type", ""))
                    content = r.content
                else:
                    content = base64.b64decode(str(b64))
            except Exception as e:
                return f"错误：获取第 {i} 张图片失败：{e}"
            path = os.path.join(out_dir, f"{prefix}{stamp}_{i}{ext}")
            try:
                with open(path, "wb") as f:
                    f.write(content)
            except Exception as e:
                return f"错误：保存第 {i} 张图片失败：{e}"
            saved.append(path)

        title = f"已生成 {len(saved)} 张图片，保存在 .bigcodex_uploads/：" if len(saved) > 1 \
            else "已生成图片，保存在 .bigcodex_uploads/："
        return title + "\n" + "\n".join(f"- {p}" for p in saved)

    @classmethod
    def _prepare_input_image(cls, context: ToolContext, image: str):
        """把输入图片转换为接口可用的值：http(s) 链接原样透传，本地文件转 base64 data URL。"""
        if image.lower().startswith(("http://", "https://")):
            return image
        try:
            resolved = context.resolve_path(image, "generate_image")
        except Exception as e:
            return f"错误：{e}"
        if not os.path.isfile(resolved):
            return f"错误：输入图片文件不存在（{resolved}）"
        mime = mimetypes.guess_type(resolved)[0] or "image/png"
        try:
            with open(resolved, "rb") as f:
                raw = f.read()
        except Exception as e:
            return f"错误：读取输入图片失败（{resolved}）：{e}"
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    @classmethod
    def _guess_ext(cls, img_url: str, content_type: str) -> str:
        """根据 URL 后缀或响应 Content-Type 推断图片扩展名，默认 .png。"""
        path = urlparse(img_url).path or ""
        suffix = os.path.splitext(path)[1].lower()
        if suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            return ".jpg" if suffix == ".jpeg" else suffix
        ctype = (content_type or "").split(";")[0].strip().lower()
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        return mapping.get(ctype, ".png")

    @classmethod
    def _clamp_int(cls, value, lo: int, hi: int, default):
        """把值钳制到 [lo, hi]；非法时返回 default。"""
        try:
            return max(lo, min(hi, int(value)))
        except (TypeError, ValueError):
            return default

    @classmethod
    def _clamp_float(cls, value, lo: float, hi: float, default: float) -> float:
        """把浮点值钳制到 [lo, hi]；非法时返回 default。"""
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return default
