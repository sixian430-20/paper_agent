"""
智能体核心：加载系统提示词、驱动智谱 GLM 工具调用调度循环、打印演示日志。

调度循环逻辑可在此文件中按需微调；系统提示词从 prompt.txt 读取。
"""

from __future__ import annotations

import json
from typing import Any

from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

from config import (
    MAX_TOOL_ROUNDS,
    PROMPT_FILE,
    ZHIPU_API_KEY,
    ZHIPU_BASE_URL,
    ZHIPU_MODEL,
    validate_keys,
)
from tools import LLM_TOOLS, dispatch_tool_call


# ---------------------------------------------------------------------------
# 日志辅助（演示截图用）
# ---------------------------------------------------------------------------

LOG_SEPARATOR = "=" * 72

# 工具编号（控制台展示用）
TOOL_DISPLAY_IDS: dict[str, str] = {
    "read_pdf_paper": "1",
    "search_academic_literature": "2",
}


def _format_for_display(content: Any) -> Any:
    """将结构化数据转为仅含中文键名的展示格式。"""
    if isinstance(content, dict):
        key_map = {
            "prompt_file": "提示词文件",
            "length": "字符数",
            "model": "模型",
            "tool_call_id": "调用编号",
            "tool_name": "工具编号",
            "arguments": "参数",
            "success": "成功",
            "error": "错误类型",
            "message": "说明",
            "file_path": "文件路径",
            "page_count": "页数",
            "char_count": "字符数",
            "full_text": "全文",
            "query": "检索词",
            "result_count": "结果数量",
            "papers": "文献列表",
            "title": "标题",
            "year": "年份",
            "abstract": "摘要",
            "link": "链接",
            "authors": "作者信息",
        }
        error_labels = {
            "file_not_found": "文件不存在",
            "not_a_file": "路径不是文件",
            "invalid_format": "格式无效",
            "empty_content": "内容为空",
            "read_failed": "读取失败",
            "missing_api_key": "缺少接口密钥",
            "timeout": "请求超时",
            "request_failed": "请求失败",
            "invalid_response": "响应无效",
            "serpapi_error": "检索服务错误",
            "unknown_tool": "未知工具",
            "invalid_arguments_json": "参数解析失败",
            "validation_error": "参数校验失败",
        }
        formatted: dict[str, Any] = {}
        for key, value in content.items():
            display_key = key_map.get(key, key)
            if key in {"tool_name", "工具编号"} and isinstance(value, str):
                formatted[display_key] = TOOL_DISPLAY_IDS.get(value, value)
            elif key in {"error", "错误类型"} and isinstance(value, str):
                formatted[display_key] = error_labels.get(value, value)
            elif key in {"success", "成功"} and isinstance(value, bool):
                formatted[display_key] = "是" if value else "否"
            else:
                formatted[display_key] = _format_for_display(value)
        return formatted
    if isinstance(content, list):
        return [_format_for_display(item) for item in content]
    return content


def _log(section: str, content: Any) -> None:
    """统一格式化打印日志块。"""
    print(f"\n{LOG_SEPARATOR}")
    print(f"【{section}】")
    print(LOG_SEPARATOR)
    if isinstance(content, (dict, list)):
        display_content = _format_for_display(content)
        print(json.dumps(display_content, ensure_ascii=False, indent=2))
    else:
        print(content)
    print(LOG_SEPARATOR)


def load_system_prompt() -> str:
    """从同目录 prompt.txt 读取系统提示词。"""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"未找到系统提示词文件: {PROMPT_FILE}\n"
            "请在项目目录创建 prompt.txt 并编写你的系统提示词。"
        )
    text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("prompt.txt 内容为空，请填写系统提示词。")
    return text


class PaperAgent:
    """课程论文 AI 助手：基于智谱 GLM 工具调用的智能体。"""

    def __init__(self) -> None:
        missing = validate_keys()
        if missing:
            raise EnvironmentError(
                f"缺少必要环境变量: {', '.join(missing)}。请复制 .env.example 为 .env 并填写密钥。"
            )

        self.client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)
        self.system_prompt = load_system_prompt()
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        _log("系统提示词已加载", {"提示词文件": str(PROMPT_FILE), "字符数": len(self.system_prompt)})

    def reset_conversation(self, keep_system: bool = True) -> None:
        """清空对话历史，可选保留 system 消息。"""
        if keep_system:
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def run(self, user_input: str) -> str:
        """
        执行一轮用户请求，自动处理多轮工具调用，返回最终 AI 文本回答。

        调度循环说明（可按 BYOA 实验需求在此微调）：
        1. 将用户消息追加到 messages
        2. 调用 chat.completions.create(tools=...)
        3. 若 assistant 消息含 tool_calls，则逐个 dispatch 并回填 tool 消息
        4. 重复直到无 tool_calls 或达到 MAX_TOOL_ROUNDS
        """
        self.messages.append({"role": "user", "content": user_input})
        _log("用户输入", user_input)

        final_answer = ""

        for round_index in range(1, MAX_TOOL_ROUNDS + 1):
            _log(f"模型调用 — 第 {round_index} 轮", {"模型": ZHIPU_MODEL})

            try:
                response = self.client.chat.completions.create(
                    model=ZHIPU_MODEL,
                    messages=self.messages,
                    tools=LLM_TOOLS,
                    tool_choice="auto",
                )
            except AuthenticationError:
                message = (
                    "智谱 AI 接口认证失败（401）：密钥无效或已过期。\n"
                    "请登录 https://open.bigmodel.cn/ 检查或重新生成 API 密钥，"
                    "并更新 .env 中的 ZHIPU_API_KEY 后重试。"
                )
                _log("调用失败", message)
                return message
            except RateLimitError:
                message = "智谱 AI 请求过于频繁或额度不足，请稍后重试或检查账户余额。"
                _log("调用失败", message)
                return message
            except APIConnectionError:
                message = "无法连接智谱 AI 接口，请检查网络连接后重试。"
                _log("调用失败", message)
                return message

            assistant_message = response.choices[0].message
            # 将 SDK 对象转为可序列化的 dict，便于追加到 messages
            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_message.content,
            }

            tool_calls = assistant_message.tool_calls
            if tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            self.messages.append(assistant_dict)

            if not tool_calls:
                final_answer = assistant_message.content or ""
                _log("最终 AI 回答", final_answer)
                return final_answer

            # 处理本轮所有工具调用
            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments

                _log(
                    "工具调用记录",
                    {
                        "调用编号": tc.id,
                        "工具编号": TOOL_DISPLAY_IDS.get(tool_name, tool_name),
                        "参数": tool_args,
                    },
                )

                result = dispatch_tool_call(tool_name, tool_args)

                # PDF 全文过长时，日志截断显示，但完整内容仍传给模型
                log_result = result
                if (
                    tool_name == "read_pdf_paper"
                    and result.get("成功")
                    and "全文" in result
                ):
                    log_result = {
                        **result,
                        "全文": (
                            result["全文"][:500]
                            + f"\n... [已截断，完整长度 {result['字符数']} 字符]"
                        ),
                    }

                _log("工具返回数据", log_result)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # 达到最大轮次仍未结束
        fallback = "已达到最大工具调用轮次，请缩小任务范围后重试。"
        _log("调度终止", fallback)
        return fallback
