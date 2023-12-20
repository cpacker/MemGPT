import * as z from 'zod';

export const MessageSchema = z.object({
  type: z.enum([
    'user_message',
    'agent_response',
    'system_message',
  ]),
  message_type: z.enum([
    'internal_monologue',
    'function_call',
    'function_return',
    'assistant_message',
    'user_message',
    'internal_error'
  ]).optional(),
  message: z.string().optional(),
  date: z.date()
});

export type Message = z.infer<typeof MessageSchema>;
