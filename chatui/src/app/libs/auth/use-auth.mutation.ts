import { useMutation } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';
import { useAuthStoreActions } from './auth.store';

export type AuthResponse = { uuid: string };
export const useAuthMutation = () => {
	const { setAsAuthenticated } = useAuthStoreActions();
	return useMutation({
		mutationKey: ['auth'],
		mutationFn: (password: string) =>
			fetch(API_BASE_URL + `/auth`, {
				method: 'POST',
				headers: { 'Content-Type': ' application/json' },
				body: JSON.stringify({ password }),
			}).then((res) => {
				if (!res.ok) {
					throw new Error('Network response was not ok');
				}
				return res.json();
			}) as Promise<AuthResponse>,
		onSuccess: (data, password) => setAsAuthenticated(data.uuid, password),
	});
};
