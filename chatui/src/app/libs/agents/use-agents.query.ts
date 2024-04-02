import { useQuery } from '@tanstack/react-query';
import { useAuthBearerToken } from '../auth/auth.store';
import { API_BASE_URL } from '../constants';
import { Agent } from './agent';

export const useAgentsQuery = (userId: string | null | undefined) => {
	const bearerToken = useAuthBearerToken();
	return useQuery({
		queryKey: [userId, 'agents', 'list'],
		enabled: !!userId,
		queryFn: async () =>
			(await fetch(API_BASE_URL + `/agents?user_id=${userId}`, {
				headers: {
					Authorization: bearerToken,
				},
			}).then((res) => res.json())) as Promise<{
				num_agents: number;
				agents: Agent[];
			}>,
	});
};
