"""
该文件中的NeedSpeakAgent类继承ConversableAgent
该类补充了need_speak方法，用于判断当前该agent是否需要对
"""
from typing import Callable, Dict, Literal, Optional, Union, List

from autogen.runtime_logging import log_new_agent, logging_enabled

from .conversable_agent import ConversableAgent
# from .. import Agent


class NeedSpeakAgent(ConversableAgent):
    def __init__(
            self,
            name: str,
            enable_need_speak: Optional[bool] = False,
            need_speak_method: Optional[Callable] = None,
            **kwargs
    ):
        """
        Args:
            name (str): name of the agent.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                    the number of auto reply reaches the max_consecutive_auto_reply.
                (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
            function_map (dict[str, callable]): Mapping function names (passed to openai) to callable functions.
            code_execution_config (dict or False): config for the code execution.
                To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:
                - work_dir (Optional, str): The working directory for the code execution.
                    If None, a default working directory will be used.
                    The default working directory is the "extensions" directory under
                    "path_to_autogen".
                - use_docker (Optional, list, str or bool): The docker image to use for code execution.
                    Default is True, which means the code will be executed in a docker container. A default list of images will be used.
                    If a list or a str of image name(s) is provided, the code will be executed in a docker container
                    with the first image successfully pulled.
                    If False, the code will be executed in the current environment.
                    We strongly recommend using docker for code execution.
                - timeout (Optional, int): The maximum execution time in seconds.
                - last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1.
            default_auto_reply (str or dict or None): the default auto reply message when no code execution or llm based reply is generated.
            llm_config (dict or False or None): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
                for available options.
                Default to False, which disables llm-based auto reply.
                When set to None, will use self.DEFAULT_CONFIG, which defaults to False.
            system_message (str or List): system message for ChatCompletion inference.
                Only used when llm_config is not False. Use it to reprogram the agent.
            description (str): a short description of the agent. This description is used by other agents
                (e.g. the GroupChatManager) to decide when to call upon this agent. (Default: system_message)
        """
        super().__init__(
            name,
            **kwargs
        )
        self.enable_need_speak = enable_need_speak
        self._need_speak_method = need_speak_method

        if logging_enabled():
            log_new_agent(self, locals())

    def need_speak(self, to: Optional[ConversableAgent] = None) -> bool:
        """消息上下文 + 自己的人设（description） + prompt"""
        """分层"""
        if not self.enable_need_speak:
            return False

        if self._need_speak_method is not None:
            return self._need_speak_method(self, to)

        SYSTEM_TEMPLATE = """
你正在参与一个角色扮演游戏。你在游戏中的角色名字是{agent_name},你的人设是
{description}
理解下面的对话，然后判断你是否需要作为下一个发言人发言，使讨论更加热烈有效。只返回True或False，不要提供原因。)
        """

        '''
        获取该agent的消息上下文
        '''
        # e.g.拼接后发送给llm的消息的例子
        # messages = [
        #     {"role": "system", "content": SYSTEM_TEMPLATE},
        #     {"role": "user", name: "li", "content": "Hello!"},
        #     {"role": "user", name: "wang", "content": "Hello!"},
        #     {"role": "user", name: "zhang", "content": "Hello!"},
        # ]
        message_context = None
        if to is None:
            n_conversations = len(self._oai_messages)
            if n_conversations == 0:
                return False
            elif n_conversations == 1:
                for conversation in self._oai_messages.values():
                    message_context = conversation.copy()
                    break
            else:
                raise ValueError(
                    "More than one conversation is found. Please specify the sender to get the last message.")
        elif to not in self._oai_messages.keys():
            raise KeyError(
                f"The agent '{to.name}' is not present in any conversation. No history available for this agent."
            )
        else:
            message_context = self._oai_messages.get(to).copy()

        """
        拼接上下文
        """
        message_context = ([{
            "role": "system",
            "content": SYSTEM_TEMPLATE.format(agent_name=self.name, description=self.description)
        }] + message_context)

        response = self._generate_oai_reply_from_client(llm_client=self.client, messages=message_context,
                                                        cache=None)
        if response.lower().find("false"):
            return False

        return True
