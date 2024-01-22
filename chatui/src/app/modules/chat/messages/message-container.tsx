import { Badge } from '@memgpt/components/badge';
import { useEffect, useRef } from 'react';
import { Agent } from '../../../libs/agents/agent';
import { Message } from '../../../libs/messages/message';
import { ReadyState } from '../../../libs/messages/message-stream.store';
import MessageContainerLayout from './message-container-layout';
import { pickMessageElement } from './message/pick-message-element';
import SelectAgentForm from './select-agent-form';
import ThinkingIndicator from './thinking-indicator';

const MessageContainer = ({
	currentAgent,
	messages,
	readyState,
	previousMessages,
}: {
	currentAgent: Agent | null;
	messages: Message[];
	readyState: ReadyState;
	previousMessages: { role: string; content: string; name: string; function_call: { arguments: string } }[];
}) => {
	const messageBox = useRef<HTMLDivElement>(null);
	useEffect(() => messageBox.current?.scrollIntoView(false), [messages]);

	if (!currentAgent) {
		return (
			<MessageContainerLayout>
				<SelectAgentForm />
			</MessageContainerLayout>
		);
	}

	return (
		<MessageContainerLayout>
			<Badge
				className="sticky left-1/2 top-2 z-10 mx-auto origin-center -translate-x-1/2 bg-background py-1 px-4"
				variant="outline"
			>
				<span>{currentAgent.name}</span>
			</Badge>
			<div className="flex flex-1 flex-col space-y-4 px-4 py-6" ref={messageBox}>
				{previousMessages.map((m) => (
					<p>
						{m.name} | {m.role} | {m.content} | {m.function_call?.arguments}
					</p>
				))}
				{messages.map((message, i) => pickMessageElement(message, i))}
				{readyState === ReadyState.LOADING ? <ThinkingIndicator className="flex items-center py-3 px-3" /> : undefined}
			</div>
		</MessageContainerLayout>
	);
};

export default MessageContainer;
