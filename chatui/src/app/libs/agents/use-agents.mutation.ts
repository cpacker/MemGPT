import { useMutation, useQueryClient } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';
import { Agent } from './agent';

export const useAgentsCreateMutation = () => {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn: async (params: {
			user_id: string;
			config: { name: string; human: string; persona: string; model: string };
		}) =>
			(await fetch(API_BASE_URL + '/agents', {
				method: 'POST',
				headers: { 'Content-Type': ' application/json' },
				body: JSON.stringify(params),
			}).then((res) => res.json())) as Promise<Agent>,
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ['agents'] }),
	});
};
