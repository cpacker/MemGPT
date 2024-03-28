import { useQuery } from '@tanstack/react-query';
import { useAuthBearerToken } from '../auth/auth.store';
import { API_BASE_URL } from '../constants';
import { AgentMemory } from './agent-memory';

export const useAgentMemoryQuery = (userId: string | null | undefined, agentId: string | null | undefined) => {
	const bearerToken = useAuthBearerToken();
	return useQuery({
		queryKey: [userId, 'agents', 'entry', agentId, 'memory'],
		queryFn: async () =>
			(await fetch(API_BASE_URL + `/agents/${agentId}/memory?user_id=${userId}`, {
				headers: {
					Authorization: bearerToken,
				},
			}).then((res) => res.json())) as Promise<AgentMemory>,
		enabled: !!userId && !!agentId,
	});
};
