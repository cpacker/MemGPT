import { useQuery } from '@tanstack/react-query';
import { useAuthBearerToken } from '../auth/auth.store';
import { API_BASE_URL } from '../constants';
import { Human } from './human';

export const useHumansQuery = (userId: string | null | undefined) => {
	const bearerToken = useAuthBearerToken();
	return useQuery({
		queryKey: [userId, 'humans', 'list'],
		enabled: !!userId,
		queryFn: async () => {
			const response = await fetch(`${API_BASE_URL}/humans?user_id=${encodeURIComponent(userId || '')}`, {
				headers: {
					Authorization: bearerToken,
				},
			});
			if (!response.ok) {
				throw new Error('Network response was not ok for fetching humans');
			}
			return (await response.json()) as Promise<{
				humans: Human[];
			}>;
		},
	});
};
