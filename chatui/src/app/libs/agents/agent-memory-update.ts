import { z } from 'zod';

export const AgentMemoryUpdateSchema = z.object({
	persona: z.string(),
	human: z.string(),
	user_id: z.string()
});

export type AgentMemoryUpdate = z.infer<typeof AgentMemoryUpdateSchema>;
