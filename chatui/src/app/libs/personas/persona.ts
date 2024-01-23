import { z } from 'zod';

export const PersonaSchema = z.object({
	name: z.string(),
	text: z.string(),
});

export type Persona = z.infer<typeof PersonaSchema>;
