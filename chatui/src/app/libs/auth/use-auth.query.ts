import { useQuery } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';

export type AuthResponse = { uuid: string };
export const useAuthQuery = () =>
	useQuery({
		queryKey: ['auth'],
		queryFn: async () => (await fetch(API_BASE_URL + `/auth`).then((res) => res.json())) as Promise<AuthResponse>,
	});
