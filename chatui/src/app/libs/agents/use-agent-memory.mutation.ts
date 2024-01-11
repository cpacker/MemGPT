import { useMutation, useQueryClient } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';
import { AgentMemoryUpdate } from './agent-memory-update';

export const useAgentMemoryUpdateMutation = () => {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn: async (params: AgentMemoryUpdate) =>
			await fetch(API_BASE_URL + `/agents/memory`, {
				method: 'POST',
				headers: { 'Content-Type': ' application/json' },
				body: JSON.stringify(params),
			}).then((res) => res.json()),
		onSuccess: (res, { agent_id }) => queryClient.invalidateQueries({ queryKey: ['agents', agent_id, 'memory'] }),
	});
};
