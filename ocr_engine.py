"""
全局 PaddleOCR 单例
===================
懒加载 + 线程安全：首次调用时创建 PaddleOCR 实例并复用，
避免每次识别都重新加载模型。模型加载较重，server 启动时在后台线程预热。
使用本地 tiny 模型（默认 <应用根>/models/ocr/，可用环境变量覆盖），不联网下载。
"""
import os
import threading
from typing import Any, List, Optional

from path_util import strip_vermagic

# 应用根目录（本文件位于 <root>/src/agent/，安装包内模型置于 <root>/models/ocr/）
_APP_ROOT = strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _resolve_model_dir(env_name: str, model_name: str) -> str:
    """模型目录解析：环境变量（如 OCR_DET_MODEL_DIR）优先，否则默认 models/ocr/<model_name>。"""
    env = os.environ.get(env_name, "").strip()
    if env:
        return env.rstrip("/\\") + os.sep
    return os.path.join(_APP_ROOT, "models", "ocr", model_name)


# 本地 tiny 模型目录（模型已放置在对应目录下，不联网下载）
OCR_DET_MODEL_DIR = _resolve_model_dir("OCR_DET_MODEL_DIR", "PP-OCRv6_tiny_det")
OCR_REC_MODEL_DIR = _resolve_model_dir("OCR_REC_MODEL_DIR", "PP-OCRv6_tiny_rec")


def _create_ocr() -> Any:
    """按项目约定创建 PaddleOCR 实例（PP-OCRv6 tiny 本地模型 + CPU 加速）。"""
    print("首次加载 OCR 模型，请稍候...")
    import paddleocr
    return paddleocr.PaddleOCR(
        text_detection_model_name="PP-OCRv6_tiny_det",
        text_detection_model_dir=OCR_DET_MODEL_DIR,
        text_recognition_model_name="PP-OCRv6_tiny_rec",
        text_recognition_model_dir=OCR_REC_MODEL_DIR,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=True,  # Intel CPU 加速
        cpu_threads=4,
    )


class OCREngine:
    """PaddleOCR 全局单例"""

    _instance: Optional[Any] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> Any:
        """获取全局 OCR 实例（懒加载，线程安全）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = _create_ocr()
        return cls._instance


def get_ocr() -> Any:
    """获取全局 OCR 实例。"""
    return OCREngine.get_instance()


def extract_rec_texts(result: Any) -> List[str]:
    """从 PaddleOCR predict 结果中提取全部识别文本（兼容 dict / 对象两种形态）。"""
    texts: List[str] = []
    for res in result or []:
        rec = None
        try:
            rec = res["rec_texts"]
        except Exception:
            try:
                rec = (res.json or {}).get("res", {}).get("rec_texts") or []
            except Exception:
                rec = []
        if isinstance(rec, list):
            texts.extend(str(t) for t in rec if str(t).strip())
    return texts


def warmup_ocr() -> bool:
    """预热 OCR 模型（供后台线程调用；失败不抛出，返回是否成功）。"""
    try:
        get_ocr()
        return True
    except Exception as e:
        print(f"[OCR] 初始化失败: {e}")
        return False
