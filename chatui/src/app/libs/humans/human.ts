import { z } from 'zod';

export const HumanSchema = z.object({
    name: z.string(),
    text: z.string(),
});

export type Human = z.infer<typeof HumanSchema>;