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

stu_a_sys = """
问题解决能力: 均衡者。具备一定的逻辑思维能力，能够理解问题并尝试不同的解决方案，但速度较慢，需要他人的引导和鼓励。
合作能力: 均衡者。具备一定的沟通能力和团队合作精神，能够积极参与团队合作，但主动性不够，需要他人的引导和鼓励。
行为特征: 在讨论中能够提出自己的想法，但不够积极主动，容易受到他人影响。

请在每轮讨论中提供互动和反馈设计的建议，并与其他教师讨论如何改进这些环节。每次输出不多于50个字。
"""
stu_b_sys = """
问题解决能力: 新手。逻辑思维能力一般，需要更多的指导和帮助才能理解问题，不善于利用资源，往往依赖他人解决问题。
合作能力: 新手。沟通能力一般，不太擅长与他人合作，容易与其他成员产生冲突，缺乏团队合作精神，只关注自己的任务，不愿意帮助他人。
行为特征: 在讨论中容易情绪化，容易与其他成员产生冲突，需要他人的引导和支持。

请在每轮讨论中提出教学方法的建议，并与其他教师讨论如何改进教学流程和活动设计。每次输出不多于50个字。
"""
stu_c_sys = """
问题解决能力: 均衡者。具备一定的逻辑思维能力，能够理解问题并尝试不同的解决方案，但速度较慢，需要他人的引导和鼓励。
合作能力: 高手。具备良好的沟通能力和团队合作精神，能够积极与他人合作并承担责任，能够倾听他人的意见，并愿意与他人分享自己的想法和资源。
行为特征: 在讨论中能够积极参与，并有效地协调团队成员，推动团队完成任务。

请在每轮讨论中提供具体的资源建议，并与其他教师讨论如何优化和整合这些资源。每次输出不多于50个字。
"""
stu_d_sys = """
问题解决能力: 高手。拥有出色的逻辑思维和分析能力，能够快速理解问题并找到解决方案，善于利用各种资源，例如网络、书籍、同学等，来获取信息并解决问题。
合作能力: 新手。沟通能力一般，不太擅长与他人合作，容易与其他成员产生冲突，缺乏团队合作精神，只关注自己的任务，不愿意帮助他人。
行为特征: 在讨论中容易情绪化，容易与其他成员产生冲突，需要他人的引导和支持。

请在每轮讨论中提供相关的技术背景和细节，并提出与其他教师讨论的技术问题和解决方案。每次输出不多于50个字。
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

task = r"""
$1.\text{\textbf{ Let }}a_1,a_2,a_3,\ldots,a_{100}$​ be integers such that
$$
\frac{a_1^2+a_2^2+a_3^2+\cdots+a_{100}^2}{a_1+a_2+a_3+\cdots+a_{100}}=100.
$$ 
Determine, with proof, the maximum possible value of $a_1.$​
"""
task2 = r"""
$2.\text{\textbf{ Nine distinct positive integers summing to 74 are put into a 3× 3 grid. Simultaneously, the number}}$​
in each cell is replaced with the sum of the numbers in its adjacent cells. (Two cells are adjacent if they share an edge.) After this, exactly four of the numbers in the grid are 23. Determine, with proof, all possible numbers that could have been originally in the center of the grid.
$3.\text{  \textbf{Let }}ABC$​ be a scalene triangle and $M$​ be the midpoint of $BC.$​ Let $X$​ be the point such that
$\dot{CX}\parallel AB$​ and $\angle AMX=90^{\circ}.\bar{\text{Prove that }AM\text{ bisects }\angle BAX}$​
$\begin{array}{lll}4. & [30] & \text{Each lattice point with nonnegative coordinates is labeled with a nonnegative integer in such a}\\  & \text{way that the point }(0,0) & \text{is labeled by }0,\text{ and for every }x,y\geq0,\text{ the set of numbers labeled on the}\end{array}$​
points $(x,y),(x,y+1)$​, and $(x+1,y)$​ is $\{n,n+1,n+2\}$​ for some nonnegative integer $n.$​ Determine,
with proof, all possible labels for the point (2000,2024).
$5.\text{\textbf{ Determine, with proof, whether there exist positive integers }}x$​ and $y$​ such that $x+y,x^2+y^2$​, and
$x^3+y^3$​ are all perfect squares.
6.  Let Q be the set of rational numbers. Given a rational number $a\neq0$​, find, with proof, all functions
$f:\mathbb{Q}\to\mathbb{Q}$​ satisfying the equation
$$
f(f(x)+ay)=af(y)+x
$$ 
for all $x,y\in\mathbb{Q}.$​

$7.\text{\textbf{ Let }}ABCDEF$​ be a regular hexagon with $P$​ as a point in its interior. Prove that of the threee
values $\tan\angle APD,\tan\angle BPE$​, and $\tan\angle CPF$​, two of them sum to the third one

8. Let $P$​ be a point in the interior of quadrilateral $ABCD$​ such that the circumcircles of triangles
$PDA,PAB$​, and $PBC$​ are pairwise distinct but congruent. Let the lines $AD$​ and $BC$​ meet at $X.$​ If
$O$​ is the circumcenter of triangle $XCD$​, prove that $\bar{OP}\perp AB.$​

9.  On each cell of a $200\times200$​ grid, we place a car, which faces in one of the four cardinal directions.
In a move, one chooses a car that does not have a car immediately in front of it, and slides it one cell forward. If a move would cause a car to exit the grid, the car is removed instead. The cars are placed so that there exists a sequence of moves that eventually removes all the cars from the grid. Across all such starting configurations, determine the maximum possible number of moves to do so.
$10.\text{\textbf{  Across all polynomials }}P$​ such that $P(n)$​ is an integer for all integers $n$​, determine, with proof, all
possible values of $P(i)$​, where $i^2=-1.$​
"""
lock = threading.Lock()


def main():
    user_proxy = NeedSpeakAgent("user",
                                code_execution_config=False,
                                # human_input_mode="ALWAYS",
                                human_input_mode="NEVER",
                                # default_auto_reply="",
                                )
    # 机器学习专家
    zhang_teacher = NeedSpeakAgent(
        name="学生A",
        # system_message="你是一位机器学习领域的专家，负责提供激活函数的技术细节和算法原理。",
        llm_config=llm_config,
        system_message=stu_a_sys,
        human_input_mode="NEVER",
        enable_need_speak=True,
    )
    # 教育学专家
    li_teacher = NeedSpeakAgent(
        name="学生B",
        # system_message="你是一位教育学专家，负责从教学方法和学生理解的角度进行讨论",
        llm_config=llm_config,
        system_message=stu_b_sys,
        enable_need_speak=True,
        human_input_mode="NEVER",
    )
    # 教学资源专家
    wang_teacher = NeedSpeakAgent(
        name="学生C",
        # system_message="你专注于教学资源的准备和使用，负责确保教师和学生有充分的材料和工具支持教学。",
        llm_config=llm_config,
        system_message=stu_c_sys,
        human_input_mode="NEVER",
        enable_need_speak=True,
    )
    # 学生反馈专家
    chen_teacher = NeedSpeakAgent(
        name="学生D",
        # system_message="你专注于学生反馈和互动设计，负责确保教学内容符合学生需求，并及时调整教学策略。",
        llm_config=llm_config,
        system_message=stu_d_sys,
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
