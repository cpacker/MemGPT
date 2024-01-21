import { useMutation, useQueryClient } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';
import { Agent } from './agent';

export const useAgentsCreateMutation = (userId: string | null | undefined) => {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn: async (params: { name: string; human: string; persona: string; model: string }) =>
			(await fetch(API_BASE_URL + '/agents', {
				method: 'POST',
				headers: { 'Content-Type': ' application/json' },
				body: JSON.stringify({ config: params, user_id: userId }),
			}).then((res) => res.json())) as Promise<Agent>,
		onSuccess: () => queryClient.invalidateQueries({ queryKey: [userId, 'agents', 'list'] }),
	});
};
