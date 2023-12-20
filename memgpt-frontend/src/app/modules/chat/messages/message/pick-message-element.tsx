import { Message } from '../../../../libs/messages/message';
import UserMessage from './user-message';
import ErrorMessage from './error-message';
import MemgptMessage from './memgpt-message';
import { cnInlineCode, cnMuted } from '@memgpt/components/typography';
import React from 'react';

export const pickMessageElement = ({ type, message_type, message, date }: Message, key: number) => {
  if (type === 'user_message') {
    return <UserMessage key={key} date={date} message={message ?? ''} />;
  }
  if (type === 'agent_response' && message_type === 'internal_error') {
    return <ErrorMessage key={key} date={date} message={message ?? ''} />;
  }
  if (type === 'agent_response' && message_type === 'assistant_message') {
    return <MemgptMessage key={key} date={date} message={message ?? ''} />;
  }
  if ((type === 'agent_response' && message_type === 'function_call' && !message?.includes('send_message'))
    || (type === 'agent_response' && message_type === 'function_return' && message !== 'None')) {
    return <p key={key} className={cnInlineCode('whitespace-nowrap overflow-x-scroll bg-black text-white border rounded max-w-xl p-2 mb-2 w-fit text-xs')}>{message}</p>;
  }
  if (type === 'agent_response' && message_type === 'internal_monologue') {
    return <p key={key} className={cnMuted('max-w-xs mb-2 w-fit text-xs p-2 rounded border')}>{message}</p>;
  }
  return undefined;
};
