import { cnMuted } from '@memgpt/components/typography';
import { LucideLoader } from 'lucide-react';
import React from 'react';

export const MessageContainerConnecting = () => <p
  className={cnMuted('flex flex-col items-center justify-center p-20')}>
  <LucideLoader className='animate-spin h-8 w-8 mb-8' />
  <span>Connecting you to your agent...</span></p>;

export default MessageContainerConnecting;
