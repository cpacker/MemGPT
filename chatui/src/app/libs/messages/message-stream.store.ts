import { fetchEventSource } from '@microsoft/fetch-event-source';
import * as z from 'zod';
import { create } from 'zustand';
import { combine } from 'zustand/middleware';
import { API_BASE_URL } from '../constants';
import { Message } from './message';

export const enum ReadyState {
	IDLE,
	LOADING,
	ERROR,
}

const useMessageStreamStore = create(
	combine(
		{
			socket: null as EventSource | null,
			socketURL: null as string | null,
			readyState: ReadyState.IDLE as ReadyState,
			abortController: null as AbortController | null,
			onMessageCallback: ((message: Message) =>
				console.warn('No message callback set up. Simply logging message', message)) as (message: Message) => void,
		},
		(set, get) => ({
			actions: {
				sendMessage: ({
					userId,
					agentId,
					message,
					role,
					bearerToken,
				}: {
					userId: string;
					agentId: string;
					message: string;
					bearerToken: string;
					role?: 'user' | 'system';
				}) => {
					const abortController = new AbortController();
					set((state) => ({ ...state, abortController, readyState: ReadyState.LOADING }));
					const onMessageCallback = get().onMessageCallback;
					const onCloseCb = () => set((state) => ({ ...state, readyState: ReadyState.IDLE }));
					const onSuccessCb = () => set((state) => ({ ...state, readyState: ReadyState.IDLE }));
					const onOpenCb = () => set((state) => ({ ...state, readyState: ReadyState.LOADING }));
					const errorCb = () =>
						set((state) => {
							abortController.abort();
							return { ...state, abortController: null, readyState: ReadyState.ERROR };
						});
					void fetchEventSource(`${API_BASE_URL}/agents/${agentId}/messages`, {
						method: 'POST',
						headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', Authorization: bearerToken },
						body: JSON.stringify({
							user_id: userId,
							message,
							role: role ?? 'user',
							stream: true,
						}),
						signal: abortController.signal,
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
							const rawData = JSON.parse(event.data);
							console.log('raw data returned in streamed response', rawData);
							const parsedData = z
								.object({ internal_monologue: z.string().nullable() })
								.or(z.object({ assistant_message: z.string() }))
								.or(z.object({ function_call: z.string() }))
								.or(z.object({ function_return: z.string() }))
								.or(z.object({ internal_error: z.string() }))
								.and(
									z.object({
										date: z
											.string()
											.optional()
											.transform((isoDate) => (isoDate ? new Date(isoDate) : new Date())),
									})
								)
								.parse(rawData);

							if ('internal_monologue' in parsedData) {
								onMessageCallback({
									type: 'agent_response',
									message_type: 'internal_monologue',
									message: parsedData['internal_monologue'] ?? 'None',
									date: parsedData.date,
								});
							} else if ('assistant_message' in parsedData) {
								onMessageCallback({
									type: 'agent_response',
									message_type: 'assistant_message',
									message: parsedData['assistant_message'],
									date: parsedData.date,
								});
								onSuccessCb();
							} else if ('function_call' in parsedData) {
								onMessageCallback({
									type: 'agent_response',
									message_type: 'function_call',
									message: parsedData['function_call'],
									date: parsedData.date,
								});
							} else if ('function_return' in parsedData) {
								onMessageCallback({
									type: 'agent_response',
									message_type: 'function_return',
									message: parsedData['function_return'],
									date: parsedData.date,
								});
							} else if ('internal_error' in parsedData) {
								onMessageCallback({
									type: 'agent_response',
									message_type: 'internal_error',
									message: parsedData['internal_error'],
									date: parsedData.date,
								});
								errorCb();
							}
						},
						onclose() {
							console.log('Connection closed by the server');
							onCloseCb();
						},
						onerror(err) {
							console.log('There was an error from server', err);
							errorCb();
						},
					});
				},
				registerOnMessageCallback: (cb: (message: Message) => void) =>
					set((state) => ({ ...state, onMessageCallback: cb })),
				abortStream: () => {
					get().abortController?.abort();
					set({ ...set, abortController: null, readyState: ReadyState.IDLE });
				},
			},
		})
	)
);

export const useMessageStreamReadyState = () => useMessageStreamStore((s) => s.readyState);
export const useMessageSocketActions = () => useMessageStreamStore((s) => s.actions);
