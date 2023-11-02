import { useEffect, useState } from 'react';

export const useMessageSubscription = () => {
  console.log('use message sub');
  const [messages, setMessages] = useState<string[]>([]);
  let sendMessage = (message: string) => {
    console.log('sending message pre connection');
  };
  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:8000/api/ws');

    websocket.onopen = () => {
      sendMessage = (message: string) => {
        console.log('sending message post connection');
        setMessages((m) => [...m, message]);
        websocket.send(message);
      };
    };
    websocket.onmessage = (event: MessageEvent<string>) => {
      setMessages((m) => [...m, event.data]);
    };

    return () => {
      websocket.readyState === 1 && websocket.close();
    };
  }, [sendMessage]);
  return [messages, sendMessage] as const;
};
