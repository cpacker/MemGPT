import { z } from 'zod';

export const AgentSchema = z.object({
  name: z.string(),
  human: z.string(),
  persona: z.string(),
  create_time: z.string()
})

export type Agent = z.infer<typeof AgentSchema>;
