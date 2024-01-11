import { cnMuted } from '@memgpt/components/typography';

const ThinkingIndicator = ({ className }: { className: string }) => (
	<div className={className}>
		<span className="relative flex h-4 w-4">
			<span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75"></span>
			<span className="relative inline-flex h-4 w-4 rounded-full bg-blue-600"></span>
		</span>
		<span className={cnMuted('ml-4')}>Thinking...</span>
	</div>
);

export default ThinkingIndicator;
