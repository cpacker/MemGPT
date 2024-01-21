import { useQuery } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';
import { Agent } from './agent';

export const useAgentsQuery = (userId: string | null | undefined) =>
	useQuery({
		queryKey: [userId, 'agents', 'list'],
		enabled: !!userId,
		queryFn: async () =>
			(await fetch(API_BASE_URL + `/agents?user_id=${userId}`).then((res) => res.json())) as Promise<{
				num_agents: number;
				agents: Agent[];
			}>,
	});
