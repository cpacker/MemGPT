import { useQuery } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';
import { AgentMemory } from './agent-memory';

export const useAgentMemoryQuery = (userId: string | null | undefined, agentId: string | null | undefined) =>
	useQuery({
		queryKey: [userId, 'agents', 'entry', agentId, 'memory'],
		queryFn: async () =>
			(await fetch(API_BASE_URL + `/agents/memory?agent_id=${agentId}&user_id=${userId}`).then((res) =>
				res.json()
			)) as Promise<AgentMemory>,
		enabled: !!userId && !!agentId,
	});
