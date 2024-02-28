import { z } from 'zod';

export const AgentSchema = z.object({
	id: z.string(),
	name: z.string(),
	human: z.string(),
	persona: z.string(),
	created_at: z.string(),
});

export type Agent = z.infer<typeof AgentSchema>;
