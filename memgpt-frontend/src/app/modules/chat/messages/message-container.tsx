import React, { useEffect, useRef } from 'react';
import MessageContainerLayout from './message-container-layout';
import ThinkingIndicator from './thinking-indicator';
import { Message } from '../../../libs/messages/message';
import { ReadyState } from '../../../libs/messages/message-stream.store';
import { pickMessageElement } from './message/pick-message-element';
import SelectAgentForm from './select-agent-form';

const MessageContainer = ({ agentSet, messages, readyState }: {
  agentSet: boolean;
  messages: Message[];
  readyState: ReadyState
}) => {
  const messageBox = useRef<HTMLDivElement>(null);
  useEffect(() => messageBox.current?.scrollIntoView(false), [messages]);

  if (!agentSet) {
    return <MessageContainerLayout>
      <SelectAgentForm/>
    </MessageContainerLayout>;
  }

  return <MessageContainerLayout>
    <div className='flex flex-col flex-1 px-4 py-6 space-y-4' ref={messageBox}>
      {messages.map((message, i) => pickMessageElement(message, i))}
      {readyState === ReadyState.LOADING ? <ThinkingIndicator className='py-3 px-3 flex items-center' /> : undefined}
    </div>
  </MessageContainerLayout>;
};

export default MessageContainer;
