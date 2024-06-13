"""
该文件用于扩展千帆大模型，但qianfan接口不兼容openai，
特别是千帆中role属性的限制，导致无法应用到小组对话的场景

"""
from __future__ import annotations

import inspect
import json
import logging
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import requests
from flaml.automl.logger import logger_formatter
from pydantic import BaseModel

from autogen.cache import Cache
from autogen.io.base import IOStream
from autogen.logger.logger_utils import get_current_ts
from autogen.oai.openai_utils import OAI_PRICE1K, get_key, is_valid_api_key
from autogen.runtime_logging import log_chat_completion, log_new_client, log_new_wrapper, logging_enabled
from autogen.token_count_utils import count_token

TOOL_ENABLED = False
try:
    import openai
except ImportError:
    ERROR: Optional[ImportError] = ImportError("Please install openai>=1 and diskcache to use autogen.OpenAIWrapper.")
    OpenAI = object
    AzureOpenAI = object
else:
    # raises exception if openai>=1 is installed and something is wrong with imports
    from openai import APIError, APITimeoutError, AzureOpenAI, OpenAI
    from openai import __version__ as OPENAIVERSION
    from openai.resources import Completions
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import ChatCompletionMessage, Choice  # type: ignore [attr-defined]
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDeltaFunctionCall,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.completion import Completion
    from openai.types.completion_usage import CompletionUsage

    if openai.__version__ >= "1.1.0":
        TOOL_ENABLED = True
    ERROR = None

try:
    from autogen.oai.gemini import GeminiClient

    gemini_import_exception: Optional[ImportError] = None
except ImportError as e:
    gemini_import_exception = e

try:
    import qianfan
except ImportError:
    ERROR: Optional[ImportError] = ImportError("Please install qianfan to use baidu qianfan llm.")

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add the console handler.
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(logger_formatter)
    logger.addHandler(_ch)

LEGACY_DEFAULT_CACHE_SEED = 41
LEGACY_CACHE_DIR = ".cache"
OPEN_API_BASE_URL_PREFIX = "https://api.openai.com"

# qianfan secret key
# config = qianfan.get_config()
# config.ACCESS_KEY = ""
# config.SECRET_KEY = ""

import os
# os.environ["QIANFAN_ACCESS_KEY"]=""
# os.environ["QIANFAN_SECRET_KEY"]=""

# 使用千帆请填入密钥
# os.environ["QIANFAN_AK"] = ""
# os.environ["QIANFAN_SK"] = ""

class QianfanClient:
    """Follows the Client protocol and wraps the OpenAI client."""

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key", None)
        print(kwargs)
        # 主要是一个属性集对象
        # self._oai_client = client
        # if (
        #     not isinstance(client, openai.AzureOpenAI)
        #     and str(client.base_url).startswith(OPEN_API_BASE_URL_PREFIX)
        #     and not is_valid_api_key(self._oai_client.api_key)
        # ):
        #     logger.warning(
        #         "The API key specified is not a valid OpenAI format; it won't work with the OpenAI-hosted model."
        #     )

    def message_retrieval(
            self, response: Union[ChatCompletion, Completion]
    ) -> Union[List[str], List[ChatCompletionMessage]]:
        """Retrieve the messages from the response."""
        choices = response.choices
        if isinstance(response, Completion):
            return [choice.text for choice in choices]  # type: ignore [union-attr]

        if TOOL_ENABLED:
            return [  # type: ignore [return-value]
                (
                    choice.message  # type: ignore [union-attr]
                    if choice.message.function_call is not None or choice.message.tool_calls is not None  # type: ignore [union-attr]
                    else choice.message.content
                )  # type: ignore [union-attr]
                for choice in choices
            ]
        else:
            return [  # type: ignore [return-value]
                choice.message if choice.message.function_call is not None else choice.message.content
                # type: ignore [union-attr]
                for choice in choices
            ]

    def create(self, params: Dict[str, Any]) -> ChatCompletion:
        """Create a completion for a given config using openai's client.

        Args:
            client: The openai client.
            params: The params for the completion.

        Returns:
            The completion.
        """
        payload = self.transform_body(params)
        chat_comp = qianfan.ChatCompletion(model="ERNIE-3.5-8K")
        # chat_comp = qianfan.ChatCompletion()

        resp = chat_comp.do(payload.get("messages"), stream=payload.get("stream"), system=payload.get("system"))

        # print(resp.get("result", resp))
        if (payload.get("stream", False)):
            pass
        else:
            finish_reason = resp.get("finish_reason", "stop")
            finish_reason = "stop" if finish_reason == "normal" else finish_reason
            msg = ChatCompletionMessage(content=resp.get("result"),role="assistant")
            choices = [Choice(finish_reason=finish_reason, index=0, message=msg)]
            usage = resp.get("usage", {})
            response = ChatCompletion(
                id=resp.get("id"),
                model="chunk.model",
                created=resp.get("created"),
                object="chat.completion",
                choices=choices,
                usage=CompletionUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
            )
            return response


    def cost(self, response: Union[ChatCompletion, Completion]) -> float:
        """Calculate the cost of the response."""
        model = response.model
        if model not in OAI_PRICE1K:
            # TODO: add logging to warn that the model is not found
            logger.debug(f"Model {model} is not found. The cost will be 0.", exc_info=True)
            return 0

        n_input_tokens = response.usage.prompt_tokens if response.usage is not None else 0  # type: ignore [union-attr]
        n_output_tokens = response.usage.completion_tokens if response.usage is not None else 0  # type: ignore [union-attr]
        if n_output_tokens is None:
            n_output_tokens = 0
        tmp_price1K = OAI_PRICE1K[model]
        # First value is input token rate, second value is output token rate
        if isinstance(tmp_price1K, tuple):
            return (tmp_price1K[0] * n_input_tokens + tmp_price1K[
                1] * n_output_tokens) / 1000  # type: ignore [no-any-return]
        return tmp_price1K * (n_input_tokens + n_output_tokens) / 1000  # type: ignore [operator]

    @staticmethod
    def get_usage(response: Union[ChatCompletion, Completion]) -> Dict:
        return {
            "prompt_tokens": response.usage.prompt_tokens if response.usage is not None else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage is not None else 0,
            "total_tokens": response.usage.total_tokens if response.usage is not None else 0,
            "cost": response.cost if hasattr(response, "cost") else 0,
            "model": response.model,
        }

    def transform_body(self, params: Dict) -> Dict:
        messages = params.get("messages", [])
        stream = params.get("stream", False)
        system = ""
        if (len(messages) > 0 and messages[0].get("role", "") == "system"):
            msg = messages.pop(0)
            system = msg.get("content", "")

        msg_length = len(messages)
        if (msg_length % 2 == 0):
            messages.insert(0, {"role": "user", "content": " "})
        return {"stream": stream, "system": system, "messages": messages}
