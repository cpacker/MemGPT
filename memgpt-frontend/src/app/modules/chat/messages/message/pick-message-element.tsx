import { Message } from '../../../../libs/messages/message';
import UserMessage from './user-message';
import ErrorMessage from './error-message';
import MemgptMessage from './memgpt-message';
import { cnMuted } from '@memgpt/components/typography';
import React from 'react';

export const pickMessageElement = ({ type, message_type, message }: Message, key: number) => {
  if (type === 'user_message') {
    return <UserMessage key={key} date={new Date()} message={message ?? ''} />;
  }
  if (type === 'agent_response_error') {
    return <ErrorMessage key={key} date={new Date()} message={message ?? ''} />;
  }
  if (type === 'agent_response' && message_type === 'assistant_message') {
    return <MemgptMessage key={key} date={new Date()} message={message ?? ''} />;
  }
  if (type === 'agent_response' && message_type === 'internal_monologue') {
    return <p key={key} className={cnMuted('mb-2 w-fit text-xs p-2 rounded border')}>{message}</p>;
  }
  return undefined;
};
