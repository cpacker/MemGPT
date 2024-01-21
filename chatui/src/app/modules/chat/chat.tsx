import { useCallback, useEffect, useRef } from 'react';
import { useAgentActions, useCurrentAgent, useLastAgentInitMessage } from '../../libs/agents/agent.store';
import { useAuthStoreState } from '../../libs/auth/auth.store';
import { useMessageHistoryActions, useMessagesForKey } from '../../libs/messages/message-history.store';
import {
	ReadyState,
	useMessageSocketActions,
	useMessageStreamReadyState,
} from '../../libs/messages/message-stream.store';
import MessageContainer from './messages/message-container';
import UserInput from './user-input';

const Chat = () => {
	const initialized = useRef(false);
	const auth = useAuthStoreState();
	const currentAgent = useCurrentAgent();
	const lastAgentInitMessage = useLastAgentInitMessage();
	const messages = useMessagesForKey(currentAgent?.id ?? '');

	const readyState = useMessageStreamReadyState();
	const { sendMessage } = useMessageSocketActions();
	const { addMessage } = useMessageHistoryActions();
	const { setLastAgentInitMessage } = useAgentActions();

	const sendMessageAndAddToHistory = useCallback(
		(message: string, role: 'user' | 'system' = 'user') => {
			if (!currentAgent || !auth.uuid) return;
			const date = new Date();
			sendMessage({ userId: auth.uuid, agentId: currentAgent.id, message, role });
			addMessage(currentAgent.id, {
				type: role === 'user' ? 'user_message' : 'system_message',
				message_type: 'user_message',
				message,
				date,
			});
		},
		[currentAgent, auth.uuid, sendMessage, addMessage]
	);

	useEffect(() => {
		if (!initialized.current) {
			initialized.current = true;
			setTimeout(() => {
				if (!currentAgent) return null;
				if (messages.length === 0 || lastAgentInitMessage?.agentId !== currentAgent.id) {
					setLastAgentInitMessage({ date: new Date(), agentId: currentAgent.id });
					sendMessageAndAddToHistory(
						'The user is back! Lets pick up the conversation! Reflect on the previous conversation and use your function calling to send him a friendly message.',
						'system'
					);
				}
			}, 300);
		}
		return () => {
			initialized.current = true;
		};
	}, [
		currentAgent,
		lastAgentInitMessage?.agentId,
		messages.length,
		sendMessageAndAddToHistory,
		setLastAgentInitMessage,
	]);

	return (
		<div className="mx-auto max-w-screen-xl p-4">
			<MessageContainer currentAgent={currentAgent} readyState={readyState} previousMessages={[]} messages={messages} />
			<UserInput enabled={readyState !== ReadyState.LOADING} onSend={sendMessageAndAddToHistory} />
		</div>
	);
};

export default Chat;
