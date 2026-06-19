"""
工具模块：定义两个独立工具及其 Pydantic 参数模型、实现与 GLM 工具调用模式。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

from config import (
    DEFAULT_SEARCH_NUM,
    PDF_BASE_DIR,
    SERPAPI_API_KEY,
    SERPAPI_ENDPOINT,
    SERPAPI_ENGINE,
)


# ---------------------------------------------------------------------------
# Pydantic 参数模型
# ---------------------------------------------------------------------------


class ReadPdfPaperArgs(BaseModel):
    """工具1：读取本地 PDF 论文的参数。"""

    file_path: str = Field(
        ...,
        description="PDF 文件路径，可为绝对路径或相对于 papers 目录的相对路径",
    )


class SearchAcademicLiteratureArgs(BaseModel):
    """工具2：SerpAPI 学术文献检索的参数。"""

    query: str = Field(..., description="学术检索关键词，支持 author:、source: 等 Scholar 语法")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回文献条数，1-20",
    )
    year_from: int | None = Field(default=None, description="起始年份（含），如 2018")
    year_to: int | None = Field(default=None, description="结束年份（含），如 2024")


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def _resolve_pdf_path(file_path: str) -> Path:
    """将用户传入路径解析为绝对 Path。"""
    path = Path(file_path)
    if path.is_absolute():
        return path
    return (PDF_BASE_DIR / path).resolve()


def read_pdf_paper(file_path: str) -> dict[str, Any]:
    """
    工具1：读取本地 PDF 并提取全文文字。

    返回统一结构，便于日志打印与模型消费。
    """
    resolved = _resolve_pdf_path(file_path)

    if not resolved.exists():
        return {
            "成功": False,
            "错误类型": "文件不存在",
            "说明": f"文件不存在：{resolved}",
            "文件路径": str(resolved),
        }

    if not resolved.is_file():
        return {
            "成功": False,
            "错误类型": "路径不是文件",
            "说明": f"路径不是文件：{resolved}",
            "文件路径": str(resolved),
        }

    if resolved.suffix.lower() != ".pdf":
        return {
            "成功": False,
            "错误类型": "格式无效",
            "说明": f"仅支持 PDF 格式，当前后缀：{resolved.suffix}",
            "文件路径": str(resolved),
        }

    try:
        reader = PdfReader(str(resolved))
        pages: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(text)

        full_text = "\n".join(pages).strip()
        if not full_text:
            return {
                "成功": False,
                "错误类型": "内容为空",
                "说明": "PDF 未能提取到可识别文字（可能为扫描件或图片型 PDF）",
                "文件路径": str(resolved),
                "页数": len(reader.pages),
            }

        return {
            "成功": True,
            "文件路径": str(resolved),
            "页数": len(reader.pages),
            "字符数": len(full_text),
            "全文": full_text,
        }

    except Exception as exc:  # noqa: BLE001 — 需将底层异常转为工具层可读信息
        return {
            "成功": False,
            "错误类型": "读取失败",
            "说明": f"读取 PDF 失败：{exc}",
            "文件路径": str(resolved),
        }


def _extract_year_from_publication_info(publication_info: dict[str, Any]) -> str | None:
    """从 SerpAPI publication_info 中尽量解析发表年份。"""
    if not publication_info:
        return None

    # 部分结果直接带 year 字段
    if "year" in publication_info and publication_info["year"]:
        return str(publication_info["year"])

    summary = publication_info.get("summary", "")
    if not summary:
        return None

    # 常见格式: "Author - Journal, 2020 - Publisher"
    match = re.search(r"\b(19|20)\d{2}\b", summary)
    return match.group(0) if match else None


def search_academic_literature(
    query: str,
    max_results: int = DEFAULT_SEARCH_NUM,
    year_from: int | None = None,
    year_to: int | None = None,
) -> dict[str, Any]:
    """
    工具2：调用 SerpAPI Google Scholar 检索学术文献。

    返回论文标题、年份、摘要（snippet）。
    """
    if not SERPAPI_API_KEY:
        return {
            "成功": False,
            "错误类型": "缺少接口密钥",
            "说明": "未配置 SerpAPI 接口密钥，请在 .env 中设置",
        }

    params: dict[str, str | int] = {
        "engine": SERPAPI_ENGINE,
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": max_results,
        "hl": "zh-CN",
    }
    if year_from is not None:
        params["as_ylo"] = year_from
    if year_to is not None:
        params["as_yhi"] = year_to

    try:
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return {
            "成功": False,
            "错误类型": "请求超时",
            "说明": "学术检索请求超时，请稍后重试",
            "检索词": query,
        }
    except requests.RequestException as exc:
        return {
            "成功": False,
            "错误类型": "请求失败",
            "说明": f"学术检索请求失败：{exc}",
            "检索词": query,
        }
    except json.JSONDecodeError:
        return {
            "成功": False,
            "错误类型": "响应无效",
            "说明": "检索服务返回了非 JSON 响应",
            "检索词": query,
        }

    if "error" in payload:
        return {
            "成功": False,
            "错误类型": "检索服务错误",
            "说明": payload.get("error", "检索服务返回错误"),
            "检索词": query,
        }

    organic = payload.get("organic_results", [])
    papers: list[dict[str, str | None]] = []

    for item in organic[:max_results]:
        publication_info = item.get("publication_info", {}) or {}
        papers.append(
            {
                "标题": item.get("title"),
                "年份": _extract_year_from_publication_info(publication_info),
                "摘要": item.get("snippet"),
                "链接": item.get("link"),
                "作者信息": publication_info.get("summary"),
            }
        )

    return {
        "成功": True,
        "检索词": query,
        "结果数量": len(papers),
        "文献列表": papers,
    }


# ---------------------------------------------------------------------------
# 智谱 GLM 工具定义（OpenAI 兼容 JSON Schema）
# ---------------------------------------------------------------------------

LLM_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_pdf_paper",
            "description": "读取本地 PDF 论文文件并提取全文文字，用于分析、摘要或引用。",
            "parameters": ReadPdfPaperArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_academic_literature",
            "description": "通过 SerpAPI 联网检索 Google Scholar 学术文献，返回标题、年份与摘要片段。",
            "parameters": SearchAcademicLiteratureArgs.model_json_schema(),
        },
    },
]

# 工具名 -> 参数模型 / 执行函数 映射，供分发器使用
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "read_pdf_paper": {
        "args_model": ReadPdfPaperArgs,
        "handler": read_pdf_paper,
    },
    "search_academic_literature": {
        "args_model": SearchAcademicLiteratureArgs,
        "handler": search_academic_literature,
    },
}


def dispatch_tool_call(tool_name: str, arguments_json: str) -> dict[str, Any]:
    """
    函数调用分发器：校验参数、执行对应工具、返回结构化结果。

    arguments_json 为模型输出的 JSON 字符串。
    """
    if tool_name not in TOOL_REGISTRY:
        return {
            "成功": False,
            "错误类型": "未知工具",
            "说明": f"未知工具：{tool_name}",
        }

    entry = TOOL_REGISTRY[tool_name]
    args_model: type[BaseModel] = entry["args_model"]
    handler = entry["handler"]

    try:
        raw_args = json.loads(arguments_json) if arguments_json else {}
        validated = args_model.model_validate(raw_args)
    except json.JSONDecodeError as exc:
        return {
            "成功": False,
            "错误类型": "参数解析失败",
            "说明": f"参数 JSON 解析失败：{exc}",
        }
    except ValidationError as exc:
        return {
            "成功": False,
            "错误类型": "参数校验失败",
            "说明": str(exc.errors()),
        }

    # 将 Pydantic 模型转为 handler 所需的关键字参数
    return handler(**validated.model_dump())
