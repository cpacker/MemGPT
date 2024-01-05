import React, { useCallback, useEffect, useRef, useState } from 'react';
import MessageContainer from './messages/message-container';
import UserInput from './user-input';
import {
  ReadyState,
  useMessageSocketActions,
  useMessageStreamReadyState,
} from '../../libs/messages/message-stream.store';
import {
  useMessageHistoryActions,
  useMessagesForKey,
} from '../../libs/messages/message-history.store';
import { useAgentActions, useCurrentAgent, useLastAgentInitMessage } from '../../libs/agents/agent.store';

const Chat = () => {
  const initialized = useRef(false);

  const currentAgent = useCurrentAgent();
  const lastAgentInitMessage = useLastAgentInitMessage();
  const messages = useMessagesForKey(currentAgent?.name ?? '');

  const readyState = useMessageStreamReadyState();
  const { sendMessage } = useMessageSocketActions();
  const { addMessage } = useMessageHistoryActions();
  const { setLastAgentInitMessage } = useAgentActions();

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const sendMessageAndAddToHistory = useCallback((message: string, role: 'user' | 'system' = 'user') => {
    if (!currentAgent) return;
    const date = new Date();
    sendMessage({ agentId: currentAgent.name, message, role });
    addMessage(currentAgent.name, {
      type: role === 'user' ? 'user_message' : 'system_message',
      message_type: 'user_message',
      message,
      date,
    });
  }, [currentAgent, sendMessage, addMessage]);

  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      setTimeout(() => {
        if (!currentAgent) return null;
        if (messages.length === 0 || lastAgentInitMessage?.agentId !== currentAgent.name) {
          setLastAgentInitMessage({ date: new Date(), agentId: currentAgent.name });
          sendMessageAndAddToHistory('The user is back! Lets pick up the conversation! Reflect on the previous conversation and use your function calling to send him a friendly message.', 'system');
        }
      }, 300);
    }
    return () => {
      initialized.current = true;
    };
  }, [currentAgent, lastAgentInitMessage?.agentId, messages.length,
    sendMessageAndAddToHistory, setLastAgentInitMessage]);

  return (<div className="max-w-screen-xl mx-auto p-4">
      <MessageContainer agentSet={!!currentAgent} readyState={readyState} messages={messages} />
      <UserInput enabled={readyState !== ReadyState.LOADING} onSend={sendMessageAndAddToHistory} />
    </div>
  );
};

export default Chat;
