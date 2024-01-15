import { z } from 'zod';

export const AgentMemorySchema = z.object({
	core_memory: z.object({
		persona: z.string(),
		human: z.string(),
	}),
	recall_memory: z.number().nullable(),
	archival_memory: z.number().nullable(),
});

export type AgentMemory = z.infer<typeof AgentMemorySchema>;
