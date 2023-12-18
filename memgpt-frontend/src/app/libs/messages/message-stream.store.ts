import { create } from 'zustand';
import { combine } from 'zustand/middleware';
import { Message } from './message';
import { fetchEventSource } from '@microsoft/fetch-event-source';

export const enum ReadyState {
  IDLE,
  LOADING,
  ERROR,
}

const ENDPOINT_URL = 'http://localhost:8283/agents/message';

const useMessageStreamStore = create(combine({
    socket: null as EventSource | null,
    agentParam: null as string | null,
    socketURL: null as string | null,
    readyState: ReadyState.IDLE,
    onMessageCallback: ((message: Message) => console.warn('No message callback set up. Simply logging message', message)) as (message: Message) => void,
  }, (set, get) => ({
    actions: {
      setAgentParam: (agentParam: string) => set(state => ({ ...state, agentParam })),
      sendMessage: (message: string) => {
        const agent_id = get().agentParam;
        const onMessageCallback = get().onMessageCallback;
        const onSuccessCb = () => set(state => ({ ...state, readyState: ReadyState.IDLE }))
        const onOpenCb = () => set(state => ({ ...state, readyState: ReadyState.LOADING }))
        const errorCb = () => set(state => ({ ...state, readyState: ReadyState.ERROR }))
        void fetchEventSource(ENDPOINT_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
          body: JSON.stringify({
            user_id: 'null',
            agent_id: agent_id,
            message,
            stream: true,
          }),
          onopen: async (res) => {
            if (res.ok && res.status === 200) {
              console.log('Connection made ', res);
              onOpenCb();
            } else if (res.status >= 400 && res.status < 500 && res.status !== 429) {
              console.log('Client-side error ', res);
              errorCb();
            }
          },
          onmessage: async (event) => {
            const parsedData = JSON.parse(event.data);

            console.log('data returned in streamed response', parsedData)

            if (parsedData['internal_monologue'] != null) {
              onMessageCallback({
                type: 'agent_response',
                message_type: 'internal_monologue',
                message: parsedData['internal_monologue'],
              })
            } else if (parsedData['assistant_message'] != null) {
              onMessageCallback({
                type: 'agent_response',
                message_type: 'assistant_message',
                message: parsedData['assistant_message'],
              })
            } else if (parsedData['function_call'] != null) {
              onMessageCallback({
                type: 'agent_response',
                message_type: 'function_call',
                message: parsedData['function_call'],
              })
            } else if (parsedData['function_return'] != null) {
              onMessageCallback({
                type: 'agent_response',
                message_type: 'function_return',
                message: parsedData['function_return'],
              })
            }
            onSuccessCb();
          },
          onclose() {
            console.log('Connection closed by the server');
          },
          onerror(err) {
            console.log('There was an error from server', err);
            errorCb();
          },
        });
      },
      resetSocket: () => set(state => {
        state?.socket?.close();
        return { ...state, socket: null };
      }),
      registerOnMessageCallback: (cb: (message: Message) => void) => set(state => ({ ...state, onMessageCallback: cb })),
    },
  }),
));

export const useMessageStreamReadyState = () => useMessageStreamStore(s => s.readyState);
export const useMessageSocketActions = () =>
  useMessageStreamStore((s) => s.actions);
