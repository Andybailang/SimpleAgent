"""
extract_from_urls 工具：从多个网页 URL 提取正文与图片链接。

纯提取工具，不含任何搜索逻辑：
- fetch_html(url)              抓取网页 HTML（浏览器 UA + 超时）
- extract_body(html, max_chars) 提取正文纯文本（剔除 script/style/nav/header/footer 等）
- extract_images_from_html(html, base_url) 提取正文图片完整 URL
- extract_from_urls(urls, extract_images, max_body_chars) 主函数，组装 JSON

图片筛选策略（多级分级，优先新闻主图而非小图标）：
1. 整页过滤：视频站点（youtube/bilibili/抖音/快手/优酷/爱奇艺等）整页跳过图片提取，
   视频页封面不是正文图；封面 URL 特征（ytimg/hdslb/bfs vpic 等）直接剔除；
2. 主图元数据优先：<head> 里的 og:image / twitter:image、JSON-LD 的 image 字段；
3. 正文容器内大图：<img srcset> / <picture><source srcset> 取最大档、懒加载 data-src 兜底；
4. 尺寸过滤：URL 显式尺寸提示或 DOM width/height/style 面积小于阈值（默认 400px）剔除；
5. 排序截断：主图元数据 > 大图 > 普通正文图，去重后最多返回 10 张。

返回 JSON 数组：每项 {url, body_text, images, status}。
"""
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext

# 抓取超时（秒）
FETCH_TIMEOUT = 20.0
# 默认每篇正文最大字符数
DEFAULT_MAX_BODY_CHARS = 3000
# 单 URL 最多提取图片数
MAX_IMAGES = 10
# 浏览器 UA（避免被简单反爬拦截）
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# 主图尺寸过滤阈值（宽或高低于该值即视为小图剔除，单位为 px）
MIN_IMG_SIZE = 400

# 正文中剔除的无用标签
_STRIP_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form",
               "noscript", "iframe", "svg")
# 图片 URL 过滤：常见小图标/logo/占位图（黑名单仅用于剔除，不做唯一筛选手段）
_IMAGE_BAD_PATTERNS = (
    "logo", "icon", "favicon", "avatar", "sprite", "placeholder",
    "spacer", "blank", "pixel", "1x1", "loading", "badge", "emoji",
    "advert", "banner", "button", "bg-", "-bg", "thumb", "thumbnail",
    "mini", "small", "btn", "icon-", "-icon", "head", "sprite-",
    # 默认头像/占位人像：虎扑 def_man、通用 default_avatar/user_face 等
    "def_man", "default_man", "default_avatar", "default_user",
    "user_avatar", "default_face", "default_profile", "default_figure",
    # 通用“头像/面部”词（很多帖子站默认头像以此命名）
    "touxiang", "face_default", "avatar_default", "man_default",
    # 百家号/百度系页面装饰图：静态 UI 图标、AI 搜索图标、头像、二维码
    "mbdp01.bdstatic.com/static",            # 静态 landing-pc 图标目录
    "psstatic.cdn.bcebos.com/basics/aichat", # AI 搜索栏图标
    "avatar.bdstatic.com",                   # 用户头像
    "/static/landing-pc/img/",              # 点赞/收藏/分享等 UI 图标
    "qrcode_",                               # 二维码
)
# 常见图片扩展名（非这些则排除）
_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp", "gif", "bmp", "svg")
# JSON-LD 主图可能出现的字段
_LD_IMAGE_KEYS = ("image", "thumbnailUrl", "contentUrl")
# 视频站域名过滤：命中则视为视频页面，整页图片跳过（视频封面不是正文图）。
# 分两组：
# - _VIDEO_HOST_EXACT：短链/单域整域精确匹配（youtu.be / b23.tv / le.com / miaopai.com 等）。
#   注意：只能用整个 hostname 精确等于匹配的域名放这里，避免后缀误伤；
# - _VIDEO_HOST_SUFFIXES：二级域后缀匹配（www.youtube.com / m.bilibili.com 等都命中）。
#   注意不要放 qq.com / sohu.com 这类泛域名，也不要放过短的二级域（如 le.com 会误伤
#   example.com 这类以 le.com 结尾的域名），只放明确独特的视频平台域。
_VIDEO_HOST_EXACT = ("youtu.be", "b23.tv", "le.com", "miaopai.com", "huoshan.com",
                     "weishi.com", "gifshow.com")
_VIDEO_HOST_SUFFIXES = (
    "youtube.com", "bilibili.com", "douyin.com", "iesdouyin.com",
    "kuaishou.com", "youku.com", "iqiyi.com",
    "v.qq.com", "weishi.qq.com", "qqlive.qq.com",
    "vimeo.com", "dailymotion.com", "tiktok.com", "ixigua.com", "1905.com",
    "tv.sohu.com",
)
# 视频封面图 URL 特征：命中则将该图片当作无意义封面剔除
# （B 站图床 i*.hdslb.com/bfs/vpic/... 用 hdslb / vpic 特征，比泛 "bili" 精准）
_VIDEO_COVER_PATTERNS = (
    "ytimg", "hqdefault", "mqdefault", "sddefault", "maxresdefault",
    "video_cover", "vthumb", "cover_video", "/cover/", "snapshot",
    "frame_", "frames_", "videocover", "video_thumb", "hdslb", "vpic",
    "thumb_video", "video_poster", "poster_image", "sp-default",
)


def _fetch_html(url: str) -> str:
    """抓取网页 HTML 源代码；失败抛异常（由上层转成 failed 状态）。

    请求头只带用户 UA，不显式发 Accept：部分站点（如百家号）会把
    ``Accept: text/html,*/*;q=0.8`` 等特征识别为爬虫并返回风控页（内容只剩
    “网络不给力”），去掉 Accept 后由 httpx 使用默认 ``*/*``，实测可正常拿到正文。
    """
    resp = httpx.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=FETCH_TIMEOUT,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text


def _is_video_page(url: str, html: str) -> bool:
    """判断网页是否属于视频站点（或页面主题为视频内容）。

    命中以下任一条件视为视频页：
    - 域名匹配视频平台（youtube/bilibili/抖音/快手/优酷/爱奇艺/腾讯视频/tiktok 等）；
    - <title>/<h1> 含“视频”类关键词（用于个别非典型域名的视频页）；
    - 页面含 <video> 标签或大量视频 iframe（如引用 bilibili/youtube 播放器）。
    """
    host = (urlparse(url).hostname or "").lower()
    if host in _VIDEO_HOST_EXACT:
        return True
    if any(host.endswith(s) for s in _VIDEO_HOST_SUFFIXES):
        return True

    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find("title")
    title = (title_el.get_text() if title_el else "") or ""
    # 标题包含明确的视频意图词（避免误伤普通新闻）
    video_title_words = ("视频", "短视频", "直播", "观看", "watch", "video",
                         "bilibili", "youtube", "tiktok", "播放")
    if any(w.lower() in title.lower() for w in video_title_words):
        return True

    # 页面内视频元素
    if soup.find("video"):
        return True
    iframes = soup.find_all("iframe")
    if len(iframes) >= 2:
        srcs = " ".join((f.get("src") or "") for f in iframes).lower()
        if "youtube" in srcs or "bilibili" in srcs or "player" in srcs:
            return True
    return False


def _is_bad_image(url: str) -> bool:
    """判断图片 URL 是否应过滤（小图标/logo/占位/视频封面等）。"""
    low = url.lower()
    path = urlparse(url).path
    # 提取扩展名：去掉 @ 之后的服务端处理参数（如 .jpeg@f_auto / .png@w_640），
    # 再取最后一段，避免把 "jpeg@f_auto" 误判为非图片格式
    last_seg = path.lower().split(".")[-1] if "." in path else ""
    ext = last_seg.split("@")[0].lower()
    # 不常见图片扩展名（非 jpg/png/webp/gif/jpeg/bmp/svg）直接排除
    if ext and ext not in _IMAGE_EXTS:
        return True
    # 视频封面特征
    if any(p in low for p in _VIDEO_COVER_PATTERNS):
        return True
    # URL 中不带扩展名（CDN 无后缀图）不直接排除，交给尺寸/来源判断，尽量保留大图
    return any(p in low for p in _IMAGE_BAD_PATTERNS)


def _resolve_url(base_url: str, raw: str) -> Optional[str]:
    """把相对 URL 补全为绝对 URL；data:/javascript: 等跳过。"""
    if not raw:
        return None
    low = raw.strip().lower()
    if low.startswith("data:") or low.startswith("javascript:"):
        return None
    try:
        return urljoin(base_url, raw.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 一、主图元数据优先（og:image / twitter:image / JSON-LD image）
# ---------------------------------------------------------------------------

def _meta_image_urls(html: str) -> List[str]:
    """收集 <head> 中的 og:image / twitter:image 主图 URL。"""
    urls: List[str] = []
    soup = BeautifulSoup(html, "lxml")
    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or meta.get("name") or "").strip().lower()
        if prop in ("og:image", "og:image:url", "og:image:secure_url",
                    "twitter:image", "twitter:image:src"):
            content = (meta.get("content") or "").strip()
            if content:
                urls.append(content)
    return urls


def _jsonld_image_urls(html: str) -> List[str]:
    """从 JSON-LD（<script type="application/ld+json">）中提取 image 字段 URL。"""
    urls: List[str] = []
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (script.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        _walk_ld(data, urls)
    return urls


def _walk_ld(node: Any, out: List[str]) -> None:
    """递归遍历 JSON-LD 对象，收集 image 相关字段的字符串 URL。"""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _LD_IMAGE_KEYS:
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
                elif isinstance(v, dict) and isinstance(v.get("url"), str):
                    out.append(v["url"].strip())
                elif isinstance(v, list):
                    for it in v:
                        if isinstance(it, str) and it.strip():
                            out.append(it.strip())
                        elif isinstance(it, dict) and isinstance(it.get("url"), str):
                            out.append(it["url"].strip())
            else:
                _walk_ld(v, out)
    elif isinstance(node, list):
        for it in node:
            _walk_ld(it, out)


def _main_image_urls(html: str, base_url: str) -> List[str]:
    """汇总主图元数据（og/twitter/JSON-LD），去重后返回绝对 URL（按出现顺序）。"""
    out: List[str] = []
    seen: set = set()
    for raw in _meta_image_urls(html) + _jsonld_image_urls(html):
        full = _resolve_url(base_url, raw)
        if full is None or full in seen or _is_bad_image(full):
            continue
        seen.add(full)
        out.append(full)
    return out


# ---------------------------------------------------------------------------
# 二、正文容器内大图（srcset / picture / 懒加载 / 普通 img）
# ---------------------------------------------------------------------------

def _srcset_max_url(srcset: str):
    """从 srcset 属性中取最大档，返回 (url, width)；无描述符时 width=None。

    按描述符宽度 w 排序取最大；全部无描述符时取最后一个 URL、width=None。
    """
    if not srcset:
        return None
    candidates = []
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split()
        url = bits[0].strip()
        if not url:
            continue
        width = None
        if len(bits) > 1:
            m = re.search(r"(\d+)w", bits[1])
            if m:
                width = int(m.group(1))
        candidates.append((width or 0, url, width))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _w, url, width = candidates[-1]
    return url, width


def _img_candidates_from_container(container: Any) -> List[Dict[str, Any]]:
    """从正文容器中收集图片候选：每条含 url、来源类型、尺寸线索、DOM 顺序。

    覆盖：<img srcset>（大图）、<picture><source srcset>、懒加载 data-src/等、
    普通 <img src>；去重由上层统一处理。
    """
    cands: List[Dict[str, Any]] = []
    seen_urls: set = set()
    order = 0
    for img in container.find_all("img"):
        url: Optional[str] = None
        kind = "img"

        # 1. srcset 最大档
        srcset_w = None
        srcset = img.get("srcset") or img.get("data-srcset") or ""
        if srcset:
            max_item = _srcset_max_url(srcset)
            if max_item:
                url, kind = max_item[0], "srcset"
                srcset_w = max_item[1]

        # 2. picture 里的 <source srcset>（当前 <img> 的父级）
        if not url:
            pic = img.find_parent("picture")
            if pic is not None:
                for src in pic.find_all("source"):
                    ss = src.get("srcset") or ""
                    if ss:
                        max_item = _srcset_max_url(ss)
                        if max_item:
                            url, kind = max_item[0], "picture"
                            srcset_w = max_item[1]
                            break

        # 3. 懒加载属性
        if not url:
            for key in ("data-src", "data-original", "data-lazy-src",
                        "data-echo", "data-url", "lazy-src"):
                v = img.get(key) or ""
                if v:
                    url, kind = v, "lazy"
                    break

        # 4. 普通 src
        if not url:
            url, kind = img.get("src") or "", "img"

        if not url or url.lower().startswith("data:"):
            continue

        # srcset/picture 来源：srcset_w 已在来源解析时取最大档宽度（w 描述符），
        # 不取 <img> 自身的 width/height（那只是 fallback 小图尺寸）

        attr_w = None
        attr_h = None
        if kind not in ("srcset", "picture"):
            attr_w = _extract_dim(img.get("width") or img.get("data-width") or "")
            attr_h = _extract_dim(img.get("height") or img.get("data-height") or "")
        style = img.get("style") or ""
        cls = img.get("class") or ""
        # 候选阶段按 URL 去重（同一张图在正文里多次引用只算一个候选；
        # 避免重复候选把更靠后的真图挤掉 / 多算一次筛选）
        if url in seen_urls:
            continue
        seen_urls.add(url)
        cands.append({
            "url": url,
            "kind": kind,
            "srcset_w": srcset_w,
            "attr_w": attr_w,
            "attr_h": attr_h,
            "style": style,
            "cls": " ".join(cls) if isinstance(cls, list) else str(cls),
            "order": order,
        })
        order += 1
    return cands


def _extract_dim(value: Any) -> Optional[int]:
    """解析 width/height 属性值为整数；非数字返回 None。"""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _style_size(style: str) -> Optional[int]:
    """从 style 里提取 width/height/min-width/max-width 像素值（取最大一个）。"""
    if not style:
        return None
    sizes: List[int] = []
    for m in re.finditer(r"(?:width|min-width|max-width)\s*:\s*(\d+(?:\.\d+)?)px", style, re.I):
        sizes.append(int(float(m.group(1))))
    if sizes:
        return max(sizes)
    return None


def _url_dim_hint(url: str) -> Optional[int]:
    """从 URL 中推测图片尺寸线索：显式尺寸段（如 /600/、-800x600）或参数（w=1200）。"""
    low = url.lower()
    # 形如 -800x600 / 800x600.jpg / /800x600/ 的路径段
    m = re.search(r"[-/_](\d{2,5})x(\d{2,5})", low)
    if m:
        return max(int(m.group(1)), int(m.group(2)))
    # 常见缩放参数 w= / width= / max-width= / size= / s= 后跟数字
    for pat in (r"[?&]w=(\d+)", r"[?&]width=(\d+)", r"[?&]h=(\d+)",
                r"[?&]height=(\d+)", r"[?&]s=(\d+)", r"[?&]size=(\d+)"):
        m = re.search(pat, low)
        if m:
            return int(m.group(1))
    return None


def _candidate_size(cand: Dict[str, Any]) -> Optional[int]:
    """计算候选图片的可用尺寸线索：srcset w 描述符、attr 宽/高、style、URL 提示中取最大值（px）。

    srcset/picture 来源优先用 srcset 最大档宽度（w 描述符），它是响应式图片的
    真实可用最大尺寸，不会被 <img> fallback 的小 width 干扰。
    """
    srcset_w = cand.get("srcset_w")
    dims = [d for d in (srcset_w, cand.get("attr_w"), cand.get("attr_h")) if d]
    style = _style_size(cand.get("style") or "")
    urld = _url_dim_hint(cand.get("url") or "")
    sizes = [d for d in dims + ([style] if style else []) + ([urld] if urld else []) if d]
    return max(sizes) if sizes else None


def _is_small_image(cand: Dict[str, Any]) -> bool:
    """判断候选是否小图（尺寸线索明确且小于阈值时剔除；无线索保留）。"""
    size = _candidate_size(cand)
    if size is None:
        return False
    return size < MIN_IMG_SIZE


# ---------------------------------------------------------------------------
# 三、正文提取
# ---------------------------------------------------------------------------

def _extract_body(html: str, max_chars: int) -> str:
    """从 HTML 提取正文纯文本，剔除无用标签。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    # 优先取 <article>/<main> 正文，否则取 body
    container = soup.find("article") or soup.find("main") or soup.body or soup
    text = container.get_text(separator="\n", strip=True)
    # 清理多余空行（保留段落间换行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


def _extract_images_from_html(html: str, base_url: str) -> List[str]:
    """提取正文图片完整 URL：主图元数据优先 + 正文大图分级 + 尺寸过滤 + 排序截断。"""
    # 视频站点/视频页的封面不是正文图，整页跳过
    if _is_video_page(base_url, html):
        return []
    soup = BeautifulSoup(html, "lxml")
    container = soup.find("article") or soup.find("main") or soup.body or soup

    # 1. 主图元数据（优先级最高，排在结果最前；去重统一由第 5 步主循环完成）
    primary: List[str] = _main_image_urls(html, base_url)
    seen: set = set()

    # 2. 正文容器内候选：<img>/srcset/picture/懒加载
    cands = _img_candidates_from_container(container)

    # 3. 尺寸过滤（阈值 MIN_IMG_SIZE=400）：明确小图剔除
    cands = [c for c in cands if not _is_small_image(c)]

    # 4. 排序：srcset/picture（大图）优先，然后按 DOM 顺序
    def _kind_rank(c: Dict[str, Any]) -> int:
        return 0 if c["kind"] in ("srcset", "picture") else 1

    cands.sort(key=lambda c: (_kind_rank(c), c["order"]))

    # 5. 去重 + 绝对化 + 黑名单剔除，主图在前，最多 MAX_IMAGES
    images: List[str] = []
    for raw in [u for u in primary] + [c["url"] for c in cands]:
        full = _resolve_url(base_url, raw)
        if full is None or full in seen or _is_bad_image(full):
            continue
        seen.add(full)
        images.append(full)
        if len(images) >= MAX_IMAGES:
            break
    return images


# ---------------------------------------------------------------------------

def _extract_from_urls(urls: List[str], extract_images: bool, max_body_chars: int) -> str:
    """主函数：循环提取每个 URL，组装 JSON 数组返回。"""
    results: List[Dict[str, Any]] = []
    for url in urls:
        item: Dict[str, Any] = {"url": url, "body_text": None, "images": [], "status": "ok"}
        try:
            html = _fetch_html(url)
            item["body_text"] = _extract_body(html, max_body_chars)
            if extract_images:
                item["images"] = _extract_images_from_html(html, url)
        except Exception as e:
            item["status"] = f"failed: {e}"
        results.append(item)
    return json.dumps(results, ensure_ascii=False)


class ExtractFromUrlsTool(BaseTool):
    """extract_from_urls 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="extract_from_urls",
            description=(
                "从多个网页 URL 提取正文纯文本与正文中的图片链接（纯提取工具，不含搜索）。"
                "参数：urls（必填，字符串数组，要提取的网页链接）；"
                "extract_images（可选，默认 true，是否提取正文图片）；"
                "max_body_chars（可选，默认 3000，每篇正文最大字符数）。"
                "返回 JSON 数组，每项含 url/body_text/images/status；"
                "status 为 ok 或 failed: 具体错误。适合在搜索后批量提取网页内容。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要提取内容的网页 URL 列表",
                    },
                    "extract_images": {
                        "type": "boolean",
                        "description": "是否提取正文中的图片链接（默认 true）",
                    },
                    "max_body_chars": {
                        "type": "integer",
                        "description": "每篇正文最多返回的字符数（默认 3000）",
                    },
                },
                "required": ["urls"],
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
        )

    @classmethod
    def execute(cls, context: ToolContext, urls: List[str],
                extract_images: bool = True, max_body_chars: int = DEFAULT_MAX_BODY_CHARS) -> str:
        if not urls:
            return "错误：urls 不能为空。"
        try:
            body_limit = max(100, min(int(max_body_chars or DEFAULT_MAX_BODY_CHARS), 20000))
        except (TypeError, ValueError):
            body_limit = DEFAULT_MAX_BODY_CHARS
        return _extract_from_urls(list(urls), bool(extract_images), body_limit)
