import React from 'react';
import { BaseMessage } from './base-message';

const UserMessage = (props: { message: string; date: Date }) => {
  return (
    <BaseMessage
      message={props.message}
      date={props.date}
      dir="rtl"
      bg="bg-muted-foreground/40"
      fg="text-black"
      initials="U"
    />
  );
};
export default UserMessage;
