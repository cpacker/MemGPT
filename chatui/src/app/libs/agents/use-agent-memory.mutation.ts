import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuthBearerToken } from '../auth/auth.store';
import { API_BASE_URL } from '../constants';
import { AgentMemoryUpdate } from './agent-memory-update';

export const useAgentMemoryUpdateMutation = (userId: string | null | undefined, agentId: string) => {
	const queryClient = useQueryClient();
	const bearerToken = useAuthBearerToken();

	return useMutation({
		mutationFn: async (params: AgentMemoryUpdate) =>
			await fetch(API_BASE_URL + `/agents/${agentId}/memory?user_id=${userId}`, {
				method: 'POST',
				headers: { 'Content-Type': ' application/json', Authorization: bearerToken },
				body: JSON.stringify(params),
			}).then((res) => res.json()),
		onSuccess: () =>
			queryClient.invalidateQueries({ queryKey: [userId, 'agents', 'entry', agentId, 'memory'] }),
	});
};
