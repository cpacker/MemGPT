import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import React, { useEffect } from 'react';
import { router } from './router';
import { Toaster } from '@memgpt/components/toast';
import { RouterProvider } from 'react-router-dom';
import { useMessageSocketActions } from './libs/messages/message-stream.store';
import { useCurrentAgent } from './libs/agents/agent.store';
import { useMessageHistoryActions } from './libs/messages/message-history.store';
import { Message } from './libs/messages/message';
import { useNextMessageLoadingActions } from './libs/messages/next-message-loading.store';
import { ThemeProvider } from './shared/theme';

const queryClient = new QueryClient();

export function App() {
  const { setAgentParam, registerOnMessageCallback, resetSocket } = useMessageSocketActions();
  const { addMessage } =useMessageHistoryActions()
  const { setLoading } = useNextMessageLoadingActions()

  const currentAgent = useCurrentAgent();

  useEffect(() => registerOnMessageCallback((message: Message) => {
    setLoading(message['type'] === 'agent_response_start');
    if (currentAgent) {
      addMessage(currentAgent.name, message);
    }
  }), [registerOnMessageCallback, currentAgent, setLoading, addMessage]);

  useEffect(() => {
    if (!currentAgent) return;
    setAgentParam(currentAgent.name);
  }, [currentAgent, setAgentParam]);

  useEffect(() => {
    return () => resetSocket();
  }, [resetSocket]);

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <RouterProvider router={router} />
        <Toaster />
      </ThemeProvider>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
