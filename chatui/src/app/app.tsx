import { Toaster } from '@memgpt/components/toast';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { useEffect } from 'react';
import { RouterProvider } from 'react-router-dom';
import { useCurrentAgent } from './libs/agents/agent.store';
import { Message } from './libs/messages/message';
import { useMessageHistoryActions } from './libs/messages/message-history.store';
import { useMessageSocketActions } from './libs/messages/message-stream.store';
import { router } from './router';
import { ThemeProvider } from './shared/theme';

const queryClient = new QueryClient();

export function App() {
	const { registerOnMessageCallback, abortStream } = useMessageSocketActions();
	const { addMessage } = useMessageHistoryActions();

	const currentAgent = useCurrentAgent();

	useEffect(() => {
		if (currentAgent) {
			abortStream();
		}
	}, [abortStream, currentAgent]);

	useEffect(
		() =>
			registerOnMessageCallback((message: Message) => {
				if (currentAgent) {
					console.log('adding message', message, currentAgent.id);
					addMessage(currentAgent.id, message);
				}
			}),
		[abortStream, registerOnMessageCallback, currentAgent, addMessage]
	);

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
