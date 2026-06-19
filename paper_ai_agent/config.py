"""
配置模块：集中管理路径、密钥与模型参数。
密钥从环境变量或同目录 .env 文件读取，不写死在代码中。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（本文件所在目录）
BASE_DIR: Path = Path(__file__).resolve().parent

# 加载 .env（若存在）
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# 智谱 AI（GLM）API 密钥与模型参数（可在 .env 中覆盖）
# 接口兼容 OpenAI SDK，默认地址：https://open.bigmodel.cn/api/paas/v4/
# ---------------------------------------------------------------------------
ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_MODEL: str = os.getenv("ZHIPU_MODEL", "glm-4-flash")
ZHIPU_BASE_URL: str = os.getenv(
    "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
)

SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")

MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "8"))

# ---------------------------------------------------------------------------
# 文件路径
# ---------------------------------------------------------------------------
PROMPT_FILE: Path = BASE_DIR / "prompt.txt"

# 本地 PDF 默认检索目录（相对路径会基于此目录解析）
PDF_BASE_DIR: Path = Path(os.getenv("PDF_BASE_DIR", str(BASE_DIR / "papers")))

# ---------------------------------------------------------------------------
# SerpAPI
# ---------------------------------------------------------------------------
SERPAPI_ENDPOINT: str = "https://serpapi.com/search"
SERPAPI_ENGINE: str = "google_scholar"
DEFAULT_SEARCH_NUM: int = int(os.getenv("SERPAPI_NUM_RESULTS", "5"))


# 环境变量缺失项的中文说明（用于控制台输出）
ENV_KEY_LABELS: dict[str, str] = {
    "ZHIPU_API_KEY": "智谱 AI 接口密钥",
    "SERPAPI_API_KEY": "SerpAPI 接口密钥",
}


def validate_keys() -> list[str]:
    """检查必要密钥是否已配置，返回缺失项中文说明列表。"""
    missing: list[str] = []
    if not ZHIPU_API_KEY:
        missing.append(ENV_KEY_LABELS["ZHIPU_API_KEY"])
    if not SERPAPI_API_KEY:
        missing.append(ENV_KEY_LABELS["SERPAPI_API_KEY"])
    return missing
