import { z } from 'zod';

export const AgentSchema = z.object({
	id: z.string(),
	name: z.string(),
	human: z.string(),
	persona: z.string(),
	created_at: z.string(),
	// TODO: Remove optional once API response returns necessary data
	memories: z.number().optional(),
	data_sources: z.number().optional(),
	last_run: z.string().optional(),
	tools: z
		.object({
			core: z.number(),
			user_defined: z.number(),
		})
		.optional(),
});

export type Agent = z.infer<typeof AgentSchema>;
