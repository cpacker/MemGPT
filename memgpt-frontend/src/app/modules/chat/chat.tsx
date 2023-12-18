import React from 'react';
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
import { useCurrentAgent } from '../../libs/agents/agent.store';

const Chat = () => {
  const currentAgent = useCurrentAgent();
  const messages = useMessagesForKey(currentAgent?.name ?? '');

  const readyState = useMessageStreamReadyState();
  const { sendMessage } = useMessageSocketActions();
  const { addMessage } = useMessageHistoryActions();

  const sendUserMessageAndAddToHistory = (message: string) => {
    if (!currentAgent) return;
    sendMessage(message);
    addMessage(currentAgent.name, {
      type: 'user_message',
      message_type: 'user_message',
      message,
    });
  };

  return (<div className='max-w-screen-xl mx-auto p-4'>
      <MessageContainer agentSet={!!currentAgent} readyState={readyState} messages={messages} />
      <UserInput enabled={readyState !== ReadyState.LOADING} onSend={sendUserMessageAndAddToHistory} />
    </div>
  );
};

export default Chat;
