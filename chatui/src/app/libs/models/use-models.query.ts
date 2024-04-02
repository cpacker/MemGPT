import { useQuery } from '@tanstack/react-query';
import { useAuthBearerToken } from '../auth/auth.store';
import { API_BASE_URL } from '../constants';
import { Model } from './model';

export const useModelsQuery = (userId: string | null | undefined) => {
	const bearerToken = useAuthBearerToken();
	return useQuery({
		queryKey: [userId, 'models', 'list'],
		enabled: !!userId,
		queryFn: async () => {
			const response = await fetch(`${API_BASE_URL}/models?user_id=${encodeURIComponent(userId || '')}`, {
				headers: {
					Authorization: bearerToken,
				},
			});
			if (!response.ok) {
				throw new Error('Network response was not ok for fetching models');
			}
			return (await response.json()) as Promise<{
				models: Model[];
			}>;
		},
	});
};
