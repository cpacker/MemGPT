import { cnInlineCode, cnMuted } from '@memgpt/components/typography';
import { Message } from '../../../../libs/messages/message';
import ErrorMessage from './error-message';
import MemgptMessage from './memgpt-message';
import UserMessage from './user-message';

export const pickMessageElement = ({ type, message_type, message, date }: Message, key: number) => {
	if (type === 'user_message') {
		return <UserMessage key={key} date={date} message={message ?? ''} />;
	}
	if (type === 'agent_response' && message_type === 'internal_error') {
		return <ErrorMessage key={key} date={date} message={message ?? ''} />;
	}
	if (type === 'agent_response' && message_type === 'assistant_message') {
		return <MemgptMessage key={key} date={date} message={message ?? ''} />;
	}
	if (
		(type === 'agent_response' && message_type === 'function_call' && !message?.includes('send_message')) ||
		(type === 'agent_response' && message_type === 'function_return' && message !== 'None')
	) {
		return (
			<p
				key={key}
				className={cnInlineCode(
					'mb-2 w-fit max-w-xl overflow-x-scroll whitespace-nowrap rounded border bg-black p-2 text-xs text-white'
				)}
			>
				{message}
			</p>
		);
	}
	if (type === 'agent_response' && message_type === 'internal_monologue') {
		return (
			<p key={key} className={cnMuted('mb-2 w-fit max-w-xs rounded border p-2 text-xs')}>
				{message}
			</p>
		);
	}
	return undefined;
};
