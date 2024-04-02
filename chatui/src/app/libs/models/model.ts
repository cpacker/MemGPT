import { z } from 'zod';

export const ModelSchema = z.object({
	name: z.string(),
	endpoint: z.string(),
	endpoint_type: z.string(),
	wrapper: z.string().optional(),
	context_window: z.number(),
});

export type Model = z.infer<typeof ModelSchema>;
