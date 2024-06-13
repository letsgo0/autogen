import json
import os
from datetime import datetime

from autogen import ConversableAgent

llm_config: dict = {"config_list": [{
    # "model": "llama3",
    "model": "qwen:7B",
    "base_url": "http://10.131.222.134:11434/v1",
    "api_type": "openai",
}]}

review_prompt = """
当前任务：{task}
完成该任务。
"""

task = """
创建一份关于机器学习激活函数章节的教学设计。
"""


def main():
    out_agent = ConversableAgent("out_agent",
                                 code_execution_config=False,
                                 human_input_mode="NEVER")

    review_agent = ConversableAgent("reviewer",
                                    human_input_mode="NEVER",
                                    llm_config=llm_config,
                                    # chat_messages={out_agent: group_chat_with_introductions.messages.copy()},
                                    )

    review_result = out_agent.initiate_chat(
        review_agent,
        cache=None,
        message=review_prompt.format(task=task),
        max_turns=1,
        clear_history=False,
    )

    filename = f'compare-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    with open(os.path.join("free_speak", filename), 'wt', encoding='utf-8') as f:
        json.dump({
            "chat": {
                "chat_history": [],
                "summary": "",
                "cost": []
            },
            "review": {
                "chat_history": review_result.chat_history,
                "summary": review_result.summary,
                "cost": review_result.cost
            }
        }, f)


if __name__ == "__main__":
    main()
