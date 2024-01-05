import React, { PropsWithChildren } from 'react';

const MessageContainerLayout = ({ children }: PropsWithChildren) =>
  <div className='relative mt-4 overflow-y-auto border bg-muted/50 rounded-md h-[70svh]'>{children}</div>;

export default MessageContainerLayout;
