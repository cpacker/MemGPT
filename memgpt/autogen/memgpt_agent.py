from autogen.agentchat import ConversableAgent, Agent
from ..agent import AgentAsync

from .. import system
from .. import constants

import asyncio
from typing import Callable, Optional, List, Dict, Union, Any, Tuple


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


class MemGPTAgent(ConversableAgent):

    def __init__(
        self,
        name: str,
        agent: AgentAsync,
        skip_verify=False
    ):
        super().__init__(name)
        self.agent = agent
        self.skip_verify = skip_verify
        self.register_reply([Agent, None], MemGPTAgent._a_generate_reply_for_user_message)
        self.register_reply([Agent, None], MemGPTAgent._generate_reply_for_user_message)

    def _generate_reply_for_user_message(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        return asyncio.run(self._a_generate_reply_for_user_message(messages=messages, sender=sender, config=config))

    async def _a_generate_reply_for_user_message(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        ret = []
        # for the interface
        self.agent.interface.reset_message_list()

        for msg in messages:
            user_message = system.package_user_message(msg['content'])
            while True:
                new_messages, heartbeat_request, function_failed, token_warning = await self.agent.step(user_message, first_message=False, skip_verify=self.skip_verify)
                ret.extend(new_messages)
                # Skip user inputs if there's a memory warning, function execution failed, or the agent asked for control
                if token_warning:
                    user_message = system.get_token_limit_warning()
                elif function_failed:
                    user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
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
        ret = {
            'role': 'assistant',
            'content': ''
        }
        lines = []
        for m in messages:
            lines.append(f"{m}")
        ret['content'] = '\n'.join(lines)
        return ret
