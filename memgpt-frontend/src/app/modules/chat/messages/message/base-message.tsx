import React from 'react';
import { Avatar, AvatarFallback, AvatarImage } from '@memgpt/components/avatar';
import * as df from 'date-fns';

export type BaseMessageProps = {
  message: string;
  date: Date;
  dir: 'ltr' | 'rtl';
  bg: string;
  fg: string;
  initials: string;
};
export const BaseMessage = (props: BaseMessageProps) => (
  <div className={`flex items-end ${props.dir === 'ltr' ? 'justify-start' : 'justify-end'}`}>
    <div className="flex flex-col space-y-1 text-xs max-w-xs mx-2 order-2 items-start">
      <div>
        <span
          className={`px-4 py-2 rounded-lg inline-block ${
            props.dir === 'ltr' ? 'rounded-bl-none' : 'rounded-br-none'
          } ${props.bg} ${props.fg}`}
        >
          {props.message}
        </span>
      </div>
      <span className="text-muted-foreground">{df.format(props.date, 'M/d/yy, h:mm a')}</span>
    </div>
    <Avatar className={props.dir === 'ltr' ? 'order-1' : 'order-2'}>
      <AvatarImage alt={props.initials} src="/placeholder.svg?height=32&width=32" />
      <AvatarFallback className="border">{props.initials}</AvatarFallback>
    </Avatar>
  </div>
);
