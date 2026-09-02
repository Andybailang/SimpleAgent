"""
PDF 处理器
==========
用户上传 PDF 时：用 PyMuPDF 把每一页转成 PNG 存入 <cwd>/.bigcodex_uploads/pdf_pages/，
再提取文字。文字类 PDF（整篇可提取文本 >= 100 字符）直接取 PyMuPDF 文本，不再走 OCR；
扫描类 PDF（整篇文本过少）仍走 PaddleOCR；混合型 PDF 中文本不足的页单独回退 OCR。
回复中额外提示每页生成的图片路径。
"""
import os
import uuid
from typing import Dict, Any, List, Optional
from .base import BaseLocalHandler
from .registry import local_handler_registry
from ocr_engine import get_ocr, extract_rec_texts

# 生成页面图片的渲染 DPI（72 的整数倍，200 兼顾清晰度与体积）
PDF_PAGE_DPI = 200
# 生成图片存放子目录（位于 <cwd>/.bigcodex_uploads/ 下）
PDF_PAGES_DIR = "pdf_pages"


def _is_pdf(att: Dict[str, Any]) -> bool:
    """判断附件是否为 PDF（按 name 优先，其次 path）。"""
    name = att.get("name") or att.get("path") or ""
    return os.path.splitext(name)[1].lower() == ".pdf"


def _resolve(cwd: str, path: str) -> str:
    """把相对/绝对路径解析到工作目录。"""
    resolved = path if os.path.isabs(path) else os.path.join(cwd, path)
    return os.path.realpath(resolved)


class PDFHandler(BaseLocalHandler):
    """PDF 转图片并提取文字"""

    name = "pdf"
    description = "PDF 转图片提取文字"

    def can_handle(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None, cwd: Optional[str] = None) -> bool:
        return any(_is_pdf(a) for a in (attachments or []))

    def handle(self, bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        """把 PDF 每页转成图片存入 .bigcodex_uploads 并 OCR，返回文字与图片路径。"""
        cwd = bs.get("cwd") or os.getcwd()
        pages_dir = os.path.join(cwd, ".bigcodex_uploads", PDF_PAGES_DIR)
        os.makedirs(pages_dir, exist_ok=True)

        pdfs = [a for a in (attachments or []) if _is_pdf(a)]
        images = [a for a in (attachments or []) if a.get("isImage") and a.get("path")]
        parts: List[str] = []
        ocr = None

        for att in pdfs:
            path = _resolve(cwd, att.get("path") or "")
            name = att.get("name") or os.path.basename(path)
            if not os.path.isfile(path):
                parts.append(f"【{name}】错误：PDF 文件不存在（{path}）")
                continue
            try:
                # PyMuPDF 1.24+ 推荐 import pymupdf，旧版本回退 fitz
                try:
                    import pymupdf as fitz
                except ImportError:
                    import fitz
                doc = fitz.open(path)
                try:
                    if doc.needs_pass:
                        parts.append(f"【{name}】错误：PDF 已加密，暂不支持提取文字")
                        continue
                    total = doc.page_count
                    # 先整体提取文本，判断 PDF 类型（文字类 / 扫描类）
                    page_texts: List[str] = []
                    for idx in range(total):
                        page_texts.append(doc.load_page(idx).get_text() or "")
                    total_text_len = sum(len(t) for t in page_texts)
                    use_ocr = total_text_len < 100  # 整篇文字过少 → 扫描类，整篇 OCR
                    page_parts: List[str] = []
                    for idx in range(total):
                        page = doc.load_page(idx)
                        zoom = PDF_PAGE_DPI / 72.0
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                        img_name = f"pdf_{uuid.uuid4().hex[:8]}_p{idx + 1}.png"
                        img_path = os.path.join(pages_dir, img_name)
                        pix.save(img_path)
                        if use_ocr or len(page_texts[idx]) < 50:
                            # 扫描页 / 混合 PDF 中文字不足的页 → OCR
                            if ocr is None:
                                ocr = get_ocr()
                            texts = extract_rec_texts(ocr.predict(img_path))
                            body = "\n".join(texts) if texts else "未识别到文字"
                        else:
                            body = page_texts[idx].strip() or "（本页无文字）"
                        page_parts.append(f"第 {idx + 1} 页（图片：{img_path}）：\n{body}")
                    kind = "转成图片并提取文字" if not use_ocr else "转成图片并用 OCR 提取文字"
                    parts.append(
                        f"【{name}】已将 {total} 页 PDF {kind}"
                        f"（图片保存在 .bigcodex_uploads/{PDF_PAGES_DIR}/）：\n"
                        + "\n\n".join(page_parts)
                    )
                finally:
                    doc.close()
            except Exception as e:
                parts.append(f"【{name}】PDF 处理失败：{e}")

        # 与 PDF 同时上传的图片附件也一并识别（延迟导入避免注册顺序被提前）
        if images:
            from .ocr import OCRHandler
            mixed = OCRHandler().handle(bs, text, images)
            if mixed:
                parts.append(mixed)

        return "\n\n".join(parts) if parts else "未发现可识别的附件"


# 导入即注册：先于 OCR 注册，含 PDF 时优先由本处理器整批处理
local_handler_registry.register(PDFHandler())
