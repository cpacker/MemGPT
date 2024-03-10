import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuthBearerToken } from '../auth/auth.store';
import { API_BASE_URL } from '../constants';
import { Persona } from './persona';

export const usePersonasCreateMutation = (userId: string | null | undefined) => {
	const queryClient = useQueryClient();
	const bearerToken = useAuthBearerToken();
	return useMutation({
		mutationFn: async (params: { name: string; text: string }): Promise<Persona> => {
			const response = await fetch(API_BASE_URL + '/agents', {
				method: 'POST',
				headers: { 'Content-Type': ' application/json', Authorization: bearerToken },
				body: JSON.stringify({ config: params, user_id: userId }),
			});

			if (!response.ok) {
				// Throw an error if the response is not OK
				const errorBody = await response.text();
				throw new Error(errorBody || 'Error creating persona');
			}

			return await response.json();
		},
		onSuccess: () => queryClient.invalidateQueries({ queryKey: [userId, 'personas', 'list'] }),
	});
};
