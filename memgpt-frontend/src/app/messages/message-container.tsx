import React, { useEffect, useRef } from 'react';
import MemgptMessage from './message/memgpt-message';
import UserMessage from './message/user-message';
import { cnMuted } from '@memgpt/components/typography';
import { Message } from '../use-messages';
import MessageContainerLayout from './message-container-layout';
import { LucideLoader } from 'lucide-react';

const getMessage = ({ role, content, function_call }: Message, key: number) => {
  if (role === 'user') {
    return <UserMessage key={key} date={new Date()} message={content} />;
  }
  if (role === 'assistant') {
    return <div key={key}>
      <p className={cnMuted('mb-2 w-fit text-xs p-2 rounded border')}>{content}</p>
      <MemgptMessage date={new Date()} message={function_call?.arguments['message']} />
    </div>;
  }
  return undefined;
};

const MessageContainer = ({ messages }: { messages: Message[] }) => {
  const messageBox = useRef<HTMLDivElement>(null);
  useEffect(() => messageBox.current?.scrollIntoView(false), [messages]);
  return <MessageContainerLayout>{
    messages.length === 0 ? (
      <p className={cnMuted('flex flex-col items-center justify-center p-20')}>
        <LucideLoader className="animate-spin h-8 w-8 mb-8"/>
        <span>Connecting you to your agent...</span></p>
    ) : (
      <div className='flex flex-col flex-1 px-4 py-6 space-y-4' ref={messageBox}>
        {messages.map((message, i) => getMessage(message, i)).filter(e => !!e)}
      </div>
    )}</MessageContainerLayout>;
};

export default MessageContainer;
