import React, { useEffect, useRef } from 'react';
import MemgptMessage from './message/memgpt-message';
import UserMessage from './message/user-message';
import { cnMuted } from '@memgpt/components/typography';
import { Message } from '../use-messages';

const getMessage = ({role, content, function_call}: Message, key: number) => {
  if (role === 'user') {
    return <UserMessage key={key} date={new Date()} message={content} />
  }
  if (role === 'assistant') {
    return <div key={key}>
      <p className={cnMuted('mb-2 w-fit text-xs p-2 rounded border')}>{content}</p>
      <MemgptMessage date={new Date()} message={function_call?.arguments['message']} />
    </div>
  }
  return undefined
}
const MessageContainer = ({ messages }: { messages: Message[] }) => {
  const messageBox = useRef<HTMLDivElement>(null);
  useEffect(() => messageBox.current?.scrollIntoView(false), [messages]);
  const messageLayout =
    messages.length === 0 ? (
      <p className={cnMuted('text-center p-20')}>Send a message to start your conversation...</p>
    ) : (
      <div className="flex flex-col flex-1 px-4 py-6 space-y-4" ref={messageBox}>
        {messages.map((message, i) => getMessage(message, i)).filter(e => !!e)}
      </div>
    );

  return <div className="mt-4 overflow-y-auto border bg-muted rounded-md h-[70svh]">{messageLayout}</div>;
};

export default MessageContainer;
