import React from 'react';
import { ReadyState } from 'react-use-websocket';
import { useMessages } from '../../hooks/messages/use-messages';
import StatusIndicator from '../../shared/layout/status-indicator';
import MessageContainer from './messages/message-container';
import UserInput from './user-input';
import { cnH1, cnH2, cnH3 } from '@memgpt/components/typography';

const Chat = () => {
  const { readyState, messageHistory, sendMessage } = useMessages();
  return (<div className="max-w-screen-xl mx-auto p-4">
      <MessageContainer readyState={readyState} messages={messageHistory}/>
      <UserInput enabled={readyState === ReadyState.OPEN} onSend={sendMessage} />
    </div>
  );
};

export default Chat;
