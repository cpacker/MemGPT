import { create } from 'zustand';
import { combine } from 'zustand/middleware';
import { Message } from './message';

export const enum ReadyState {
  CONNECTING,
  OPEN,
  CLOSING,
  CLOSED,
}

const SOCKET_URL = 'ws://localhost:8000/api/chat';

const setUpWebsocket = (
  onMessageCallback: (message: Message) => void,
  onOpenCallback: (readyState: ReadyState) => void,
  onCloseCallback: (readyState: ReadyState) => void,
  url: string | null,
): WebSocket | null => {
  if (!url) {
    return null;
  }

  const newSocket = new WebSocket(url);

  // Connection opened
  newSocket.addEventListener('open', (event) => {
    onOpenCallback(newSocket.readyState);
  });

  // Connection closed
  newSocket.addEventListener('close', (event) => {
    onCloseCallback(newSocket.readyState);
  });

  // Listen for messages
  newSocket.addEventListener('message', (event) => {
    const jsonResponse = JSON.parse(event.data);
    onMessageCallback(jsonResponse);
  });

  return newSocket;
};

const useMessageSocketStore = create(combine({
    socket: null as WebSocket | null,
    agentParam: null as string | null,
    socketURL: null as string | null,
    readyState: ReadyState.CONNECTING,
    onMessageCallback: ((message: Message) => console.warn('No message callback set up. Simply logging message', message)) as (message: Message) => void,
  }, (set, get) => ({
    actions: {
      setAgentParam: (agentParam: string) => set(state => {
        const socketURL = agentParam ? `${SOCKET_URL}?agent=${agentParam}` : null;
        const updateReadyState = (readyState: ReadyState) => set(state => ({ ...state, readyState }));
        return {
          ...state,
          agentParam,
          socketURL,
          socket: setUpWebsocket(
            state.onMessageCallback,
            updateReadyState,
            updateReadyState,
            socketURL),
        };
      }),
      sendMessage: (message: string) => get()?.socket?.send(message),
      resetSocket: () => set(state => {
        state?.socket?.close();
        return { ...state, socket: null };
      }),
      registerOnMessageCallback: (cb: (message: Message) => void) => set(state => ({ ...state, onMessageCallback: cb })),
    },
  }),
));

export const useMessageSocketReadyState = () => useMessageSocketStore(s => s.readyState);
export const useMessageSocketActions = () =>
  useMessageSocketStore((s) => s.actions);
