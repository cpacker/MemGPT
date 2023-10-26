from autogen.agentchat import ConversableAgent, Agent
from ..agent import AgentAsync

# from .. import system
# from .. import constants

import asyncio
from typing import Callable, Optional, List, Dict, Union, Any, Tuple


from .interface import AutoGenInterface
from ..persistence_manager import InMemoryStateManager
from .. import system
from .. import constants
from .. import presets
from ..personas import personas
from ..humans import humans


def create_memgpt_autogen_agent_from_config(
    name: str,
    system_message: Optional[str] = "You are a helpful AI Assistant.",
    is_termination_msg: Optional[Callable[[Dict], bool]] = None,
    max_consecutive_auto_reply: Optional[int] = None,
    human_input_mode: Optional[str] = "TERMINATE",
    function_map: Optional[Dict[str, Callable]] = None,
    code_execution_config: Optional[Union[Dict, bool]] = None,
    llm_config: Optional[Union[Dict, bool]] = None,
    default_auto_reply: Optional[Union[str, Dict, None]] = "",
):
    """
    TODO support AutoGen config workflow in a clean way with constructors
    """
    raise NotImplementedError


def create_autogen_memgpt_agent(
    autogen_name,
    preset=presets.DEFAULT,
    model=constants.DEFAULT_MEMGPT_MODEL,
    persona_description=personas.DEFAULT,
    user_description=humans.DEFAULT,
    interface=None,
    interface_kwargs={},
    persistence_manager=None,
    persistence_manager_kwargs={},
):
    interface = AutoGenInterface(**interface_kwargs) if interface is None else interface
    persistence_manager = (
        InMemoryStateManager(**persistence_manager_kwargs)
        if persistence_manager is None
        else persistence_manager
    )

    memgpt_agent = presets.use_preset(
        preset,
        model,
        persona_description,
        user_description,
        interface,
        persistence_manager,
    )

    autogen_memgpt_agent = MemGPTAgent(
        name=autogen_name,
        agent=memgpt_agent,
    )
    return autogen_memgpt_agent


class MemGPTAgent(ConversableAgent):
    def __init__(self, name: str, agent: AgentAsync, skip_verify=False):
        super().__init__(name)
        self.agent = agent
        self.skip_verify = skip_verify
        self.register_reply(
            [Agent, None], MemGPTAgent._a_generate_reply_for_user_message
        )
        self.register_reply([Agent, None], MemGPTAgent._generate_reply_for_user_message)

    def _generate_reply_for_user_message(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        return asyncio.run(
            self._a_generate_reply_for_user_message(
                messages=messages, sender=sender, config=config
            )
        )

    async def _a_generate_reply_for_user_message(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        # ret = []
        # for the interface
        # print(f"a_gen_reply messages:\n{messages}")
        self.agent.interface.reset_message_list()

        for msg in messages:
            if "name" in msg:
                user_message_raw = f"{msg['name']}: {msg['content']}"
            else:
                user_message_raw = msg["content"]
            user_message = system.package_user_message(user_message_raw)
            while True:
                (
                    new_messages,
                    heartbeat_request,
                    function_failed,
                    token_warning,
                ) = await self.agent.step(
                    user_message, first_message=False, skip_verify=self.skip_verify
                )
                # ret.extend(new_messages)
                # Skip user inputs if there's a memory warning, function execution failed, or the agent asked for control
                if token_warning:
                    user_message = system.get_token_limit_warning()
                elif function_failed:
                    user_message = system.get_heartbeat(
                        constants.FUNC_FAILED_HEARTBEAT_MESSAGE
                    )
                elif heartbeat_request:
                    user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                else:
                    break

        # Pass back to AutoGen the pretty-printed calls MemGPT made to the interface
        pretty_ret = MemGPTAgent.pretty_concat(self.agent.interface.message_list)
        return True, pretty_ret

    @staticmethod
    def pretty_concat(messages):
        """AutoGen expects a single response, but MemGPT may take many steps.

        To accommodate AutoGen, concatenate all of MemGPT's steps into one and return as a single message.
        """
        ret = {"role": "assistant", "content": ""}
        lines = []
        for m in messages:
            lines.append(f"{m}")
        ret["content"] = "\n".join(lines)
        return ret
