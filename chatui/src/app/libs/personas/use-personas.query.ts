import { useQuery } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';
import { Persona } from './persona';

export const usePersonasQuery = (userId: string | null | undefined) =>
	useQuery({
		queryKey: [userId, 'personas', 'list'],
		enabled: !!userId, // The query will not execute unless userId is truthy
		queryFn: async () => {
			const response = await fetch(`${API_BASE_URL}/personas?user_id=${encodeURIComponent(userId || '')}`);
			if (!response.ok) {
				throw new Error('Network response was not ok');
			}
			return (await response.json()) as Promise<{
				personas: Persona[];
			}>;
		},
	});
