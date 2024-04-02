import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuthBearerToken } from '../auth/auth.store';
import { API_BASE_URL } from '../constants';
import { Agent } from './agent';

export const useAgentsCreateMutation = (userId: string | null | undefined) => {
	const queryClient = useQueryClient();
	const bearerToken = useAuthBearerToken();
	return useMutation({
		mutationFn: async (params: { name: string; human: string; persona: string; model: string }): Promise<Agent> => {
			const response = await fetch(API_BASE_URL + '/agents', {
				method: 'POST',
				headers: { 'Content-Type': ' application/json', Authorization: bearerToken },
				body: JSON.stringify({ config: params, user_id: userId }),
			});

			if (!response.ok) {
				// Throw an error if the response is not OK
				const errorBody = await response.text();
				throw new Error(errorBody || 'Error creating agent');
			}

			return await response.json();
		},
		onSuccess: () => queryClient.invalidateQueries({ queryKey: [userId, 'agents', 'list'] }),
	});
};
