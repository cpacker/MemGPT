import React from 'react';
import { ReadyState } from 'react-use-websocket';
import { Badge } from '@memgpt/components/badge';

const StatusIndicator = ({ readyState }: {readyState: ReadyState}) => {
  if (readyState === ReadyState.OPEN) {
    return <Badge className="bg-emerald-600">Connected</Badge>
  }
  if (readyState === ReadyState.CLOSED) {
    return <Badge variant="destructive">Disconnected</Badge>
  }
  if (readyState === ReadyState.CONNECTING) {
    return <Badge variant="secondary">Connecting...</Badge>
  }
  return <></>
};

export default StatusIndicator;
