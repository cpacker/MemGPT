import React, { useEffect, useRef } from 'react';
import MemgptMessage from './message/memgpt-message';
import UserMessage from './message/user-message';
import { cnMuted } from '@memgpt/components/typography';

const MessageContainer = ({ messages }: { messages: string[] }) => {
  const messageBox = useRef<HTMLDivElement>(null);
  useEffect(() => messageBox.current?.scrollIntoView(false), [messages]);
  const messageLayout =
    messages.length === 0 ? (
      <p className={cnMuted('text-center p-20')}>Send a message to start your conversation...</p>
    ) : (
      <div className="flex flex-col flex-1 px-4 py-6 space-y-4" ref={messageBox}>
        {messages.map((m, i) =>
          i % 2 == 1 ? (
            <MemgptMessage key={i} date={new Date()} message={m} />
          ) : (
            <UserMessage key={i} date={new Date()} message={m} />
          )
        )}
      </div>
    );

  return <div className="mt-4 overflow-y-auto border bg-muted rounded-md h-[70svh]">{messageLayout}</div>;
};

export default MessageContainer;
