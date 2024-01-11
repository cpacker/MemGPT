import { useEffect, useRef } from 'react';
import { Message } from '../../../libs/messages/message';
import { ReadyState } from '../../../libs/messages/message-stream.store';
import MessageContainerLayout from './message-container-layout';
import { pickMessageElement } from './message/pick-message-element';
import SelectAgentForm from './select-agent-form';
import ThinkingIndicator from './thinking-indicator';

const MessageContainer = ({
	agentSet,
	messages,
	readyState,
}: {
	agentSet: boolean;
	messages: Message[];
	readyState: ReadyState;
}) => {
	const messageBox = useRef<HTMLDivElement>(null);
	useEffect(() => messageBox.current?.scrollIntoView(false), [messages]);

	if (!agentSet) {
		return (
			<MessageContainerLayout>
				<SelectAgentForm />
			</MessageContainerLayout>
		);
	}

	return (
		<MessageContainerLayout>
			<div className="flex flex-1 flex-col space-y-4 px-4 py-6" ref={messageBox}>
				{messages.map((message, i) => pickMessageElement(message, i))}
				{readyState === ReadyState.LOADING ? <ThinkingIndicator className="flex items-center py-3 px-3" /> : undefined}
			</div>
		</MessageContainerLayout>
	);
};

export default MessageContainer;
