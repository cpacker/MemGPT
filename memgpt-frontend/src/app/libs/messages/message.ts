import * as z from 'zod';

const messageSchema = z.object({
  type: z.enum([
    'user_message',
    'agent_response',
    'agent_response_start',
    'agent_response_end',
    'agent_response_error',
  ]),
  message_type: z.enum([
    'internal_monologue',
    'function_call',
    'function_return',
    'assistant_message',
    'user_message',
  ]).optional(),
  message: z.string().optional(),
});

export type Message = z.infer<typeof messageSchema>;
