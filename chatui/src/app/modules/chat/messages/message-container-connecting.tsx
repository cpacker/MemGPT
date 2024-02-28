import { cnMuted } from '@memgpt/components/typography';
import { LucideLoader } from 'lucide-react';

export const MessageContainerConnecting = () => (
	<p className={cnMuted('flex flex-col items-center justify-center p-20')}>
		<LucideLoader className="mb-8 h-8 w-8 animate-spin" />
		<span>Connecting you to your agent...</span>
	</p>
);

export default MessageContainerConnecting;
