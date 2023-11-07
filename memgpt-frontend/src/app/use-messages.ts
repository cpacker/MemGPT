import { useCallback, useEffect, useRef, useState } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import * as z from 'zod';

const SOCKET_URL = 'ws://localhost:8000/api/ws';

const messageSchema = z.object({
  role: z.enum(['user', 'assistant', 'function']),
  content: z.string(),
  function_call: z.object({
    name: z.enum(['send_message']),
    arguments: z.any()
  }).optional()
});

export type Message = z.infer<typeof messageSchema>;

export const useMessages = () => {
  const didUnmount = useRef(false);

  const [socketUrl, setSocketUrl] = useState(SOCKET_URL);
  const [messageHistory, setMessageHistory] = useState<Message[]>([]);

  const { sendMessage, lastMessage, readyState } = useWebSocket(socketUrl, {
    shouldReconnect: (e) => e.code === 1005 &&!didUnmount.current,
    reconnectAttempts: 10,
    reconnectInterval: (attemptNumber) => Math.min(Math.pow(2, attemptNumber) * 1000, 10000),
  });

  useEffect(() => {
    if (lastMessage !== null) {
      const jsonResponse = JSON.parse(lastMessage.data);
      // const parsedResponse = z.object({
      //   'new_messages': z.array(messageSchema),
      //   'time': z.string()
      // }).parse(jsonResponse);
      console.log(jsonResponse['new_messages'])
      setMessageHistory((prev) => [...prev, ...jsonResponse['new_messages']]);
    }
  }, [lastMessage, setMessageHistory]);

  const sendMessageCb = useCallback((message: string) => {
    const newMessage: Message = {
      role: 'user',
      content: message
    }
    console.log('calling', newMessage, message);

    setMessageHistory((h) => [...h, newMessage]);
    sendMessage(message);
  }, []);

  useEffect(() => {
    return () => {
      didUnmount.current = true;
    };
  }, []);

  return { readyState, sendMessage: sendMessageCb, messageHistory };
};
