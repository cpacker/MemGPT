import React, { PropsWithChildren } from 'react';

const MessageContainerLayout = ({ children }: PropsWithChildren) =>
  <div className='mt-4 overflow-y-auto border bg-muted rounded-md h-[70svh]'>{children}</div>;

export default MessageContainerLayout;
