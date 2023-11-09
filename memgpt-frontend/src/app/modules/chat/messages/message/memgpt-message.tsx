import React from 'react';
import { BaseMessage } from './base-message';

const MemgptMessage = (props: { message: string; date: Date }) => (
  <BaseMessage message={props.message} date={props.date} dir="ltr" bg="bg-blue-600" fg="text-white" initials="AI" />
);

export default MemgptMessage;
