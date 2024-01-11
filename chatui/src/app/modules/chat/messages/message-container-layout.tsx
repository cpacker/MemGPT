import { PropsWithChildren } from 'react';

const MessageContainerLayout = ({ children }: PropsWithChildren) => (
	<div className="relative mt-4 h-[70svh] overflow-y-auto rounded-md border bg-muted/50">{children}</div>
);

export default MessageContainerLayout;
