import json
import os.path
import random
import threading
from datetime import datetime
from typing import Union

from autogen import ConversableAgent, GroupChat, GroupChatManager, Agent, UserProxyAgent, NeedSpeakAgent

llm_config: dict = {"config_list": [{
    # "model": "llama3",
    "model": "qwen:7B",
    "base_url": "http://10.131.222.134:11434/v1",
    "api_type": "openai",
}]}

chen_sys = """
陈老师，作为学生反馈专家，说明在激活函数章节中如何设计互动环节和反馈机制。接下来，在整个讨论过程中，请您关注以下几点：
1. 设计课堂互动环节，提高学生参与度。
2. 提出小测试和讨论问题，以检查学生对激活函数的理解情况。
3. 建议有效的反馈机制和工具，收集并分析学生的学习反馈。

请在每轮讨论中提供互动和反馈设计的建议，并与其他教师讨论如何改进这些环节。
"""
zhang_sys = """
张老师，作为机器学习专家，介绍激活函数的定义、作用和常见类型。接下来，在整个讨论过程中，请您关注以下几点：
1. 详细解释每种常见激活函数（如ReLU、Sigmoid、Tanh）的数学公式和应用场景。
2. 提出如何在教学中深入讲解这些技术细节。
3. 确保学生在理解这些激活函数时不会感到困惑，并建议合适的例子和实践练习。

请在每轮讨论中提供相关的技术背景和细节，并提出与其他教师讨论的技术问题和解决方案。
"""
li_sys = """
李老师，作为教育学专家，说明在教学激活函数时可以采用的有效教学方法。接下来，在整个讨论过程中，请您关注以下几点：
1. 设计教学流程，结合理论讲解和实践操作。
2. 建议如何使用图形、动画和互动工具帮助学生理解抽象概念。
3. 讨论如何设计课堂活动和练习，以检验学生的理解情况。

请在每轮讨论中提出教学方法的建议，并与其他教师讨论如何改进教学流程和活动设计。
"""
wang_sys = """
王老师，作为教学资源专家，列出在讲授激活函数章节时需要准备的教学材料和资源。接下来，在整个讨论过程中，请您关注以下几点：
1. 提供和设计有效的教学材料（如PPT、视频、图形）。
2. 讨论如何利用在线模拟器和交互工具展示激活函数的效果。
3. 确保教学资源能够在课前和课后为学生提供足够的支持。

请在每轮讨论中提供具体的资源建议，并与其他教师讨论如何优化和整合这些资源。
"""
user_sys = """
你对当前的讨论做出判断，判断是否需要结束对话。判断条件如下：
1. 所有人都没有异议。
2. 完成了任务讨论。
"""
review_prompt = """
当前任务：{task}
理解上面的对话，完成该任务。
"""
task = """
创建一份关于机器学习激活函数章节的教学设计。
"""
lock = threading.Lock()


def main():
    user_proxy = NeedSpeakAgent("user",
                                code_execution_config=False,
                                human_input_mode="NEVER",
                                # default_auto_reply="",
                                )
    # 机器学习专家
    zhang_teacher = NeedSpeakAgent(
        name="张老师",
        # system_message="你是一位机器学习领域的专家，负责提供激活函数的技术细节和算法原理。",
        llm_config=llm_config,
        system_message=zhang_sys,
        human_input_mode="NEVER",
        enable_need_speak=True,
    )
    # 教育学专家
    li_teacher = NeedSpeakAgent(
        name="李老师",
        # system_message="你是一位教育学专家，负责从教学方法和学生理解的角度进行讨论",
        llm_config=llm_config,
        system_message=li_sys,
        enable_need_speak=True,
        human_input_mode="NEVER",
    )
    # 教学资源专家
    wang_teacher = NeedSpeakAgent(
        name="王老师",
        # system_message="你专注于教学资源的准备和使用，负责确保教师和学生有充分的材料和工具支持教学。",
        llm_config=llm_config,
        system_message=wang_sys,
        human_input_mode="NEVER",
        enable_need_speak=True,
    )
    # 学生反馈专家
    chen_teacher = NeedSpeakAgent(
        name="陈老师",
        # system_message="你专注于学生反馈和互动设计，负责确保教学内容符合学生需求，并及时调整教学策略。",
        llm_config=llm_config,
        system_message=chen_sys,
        human_input_mode="NEVER",
        enable_need_speak=True,
    )

    group_chat_with_introductions = GroupChat(
        agents=[zhang_teacher, li_teacher, wang_teacher, chen_teacher, user_proxy],
        messages=[],
        max_round=10,
        # send_introductions=True,
        # select_speaker_auto_verbose=True,
        speaker_selection_method=free_speak_pattern,
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat_with_introductions,
        llm_config=llm_config,
    )

    chat_result = user_proxy.initiate_chat(
        group_chat_manager,
        message=task,
        summary_method="reflection_with_llm",
        # cache=None,
    )

    out_agent = ConversableAgent("out_agent",
                                 code_execution_config=False,
                                 human_input_mode="NEVER")

    review_agent = ConversableAgent("reviewer",
                                    human_input_mode="NEVER",
                                    llm_config=llm_config,
                                    chat_messages={out_agent: group_chat_with_introductions.messages.copy()},
                                    )

    review_result = out_agent.initiate_chat(
        review_agent,
        cache=None,
        message=review_prompt.format(task=task),
        max_turns=1,
        clear_history=False,
    )

    filename = f'free_speak-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    with open(os.path.join("free_speak", filename), 'wt', encoding='utf-8') as f:
        json.dump({
            "chat": {
                "chat_history": chat_result.chat_history,
                "summary": chat_result.summary,
                "cost": chat_result.cost
            },
            "review": {
                "chat_history": review_result.chat_history,
                "summary": review_result.summary,
                "cost": review_result.cost
            }
        }, f)


def free_speak_pattern(last_speaker: Agent, groupchat: GroupChat) -> Union[Agent, str, None]:
    #     发言意向收集
    speaker_list = []
    prepared_agent_list = []
    for agent in groupchat.agents:
        if isinstance(agent, NeedSpeakAgent):
            prepared_agent_list.append(agent)


    thread_list = [threading.Thread(
        target=_free_speak_pattern_thread,
        kwargs={
            "speaker_list": speaker_list,
            "agent": agent
        }
    ) for agent in prepared_agent_list]

    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()

    if len(speaker_list) == 0:
        print("无人发言，默认随机!")
        return random.choice(prepared_agent_list)

    #     随机返回一个
    return random.choice(speaker_list)


def _free_speak_pattern_thread(speaker_list=None, agent: ConversableAgent=None):
    if speaker_list is None:
        speaker_list = []
    flag = agent.need_speak()
    if flag:
        with lock:
            print(f'previous speaker list: {[agent.name for agent in speaker_list]}, add new agent: {agent.name}')
            speaker_list.append(agent)


if __name__ == "__main__":
    main()
