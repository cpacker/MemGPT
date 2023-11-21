import React from 'react';
import { Badge } from '@memgpt/components/badge';
import { ReadyState } from '../../../libs/messages/message-socket.store';

const twPos = (extraClasses?: string) => 'absolute h-4 w-4 p-0 top-4 right-4 ' + (extraClasses ?? '');
const StatusIndicator = ({ readyState }: {readyState: ReadyState}) => {
  if (readyState === ReadyState.OPEN) {
    return <Badge className={twPos('hover:bg-emerald-600 bg-emerald-500')}/>
  }
  if (readyState === ReadyState.CLOSED) {
    return <Badge className={twPos()} variant="destructive"/>
  }
  if (readyState === ReadyState.CONNECTING) {
    return <Badge className={twPos('animate-pulse')} variant="outline"/>
  }
  return <></>
};

export default StatusIndicator;
