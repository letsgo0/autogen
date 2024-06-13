import os

from autogen import ConversableAgent


def main():
    # llm_config: dict = dict([("config_list", "asdf")])

    llm_config: dict = {"config_list": [{
            "model": "llama3",
            "base_url": "http://10.131.222.134:11434/v1",
            "api_type": "openai",
    }]}

    agent_with_number = ConversableAgent(
        "agent_with_number",
        system_message="You are playing a game of guess-my-number. You have the "
                       "number 53 in your mind, and I will try to guess it. "
                       "If I guess too high, say 'too high', if I guess too low, say 'too low'. ",
        llm_config={"config_list": [{"api_type": "qianfan","model": "xxx", "api_key": "os.enviro"}]},
        # llm_config={"config_list": [{"model": "gpt-4", "api_key": "os.enviro"}]},
        # llm_config=llm_config,
        is_termination_msg=lambda msg: "53" in msg["content"],  # terminate if the number is guessed by the other agent
        human_input_mode="NEVER",  # never ask for human input
    )

    agent_guess_number = ConversableAgent(
        "agent_guess_number",
        system_message="I have a number in my mind, and you will try to guess it. "
                       "If I say 'too high', you should guess a lower number. If I say 'too low', "
                       "you should guess a higher number. ",
        llm_config={"config_list": [{"api_type": "qianfan", "model": "xxx", "api_key": "os.enviro"}]},
        # llm_config={"config_list": [{"model": "gpt-4", "api_key": "os.enviro"}]},
        # llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3
    )

    result = agent_with_number.initiate_chat(
        agent_guess_number,
        message="I have a number between 1 and 100. Guess it!",
    )

    print(result)



if __name__ == "__main__":
    main()
