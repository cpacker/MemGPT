import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type AuthState = {
	uuid: string | null;
	token: string | null;
	loggedIn: boolean;
};
export type AuthActions = {
	setAsAuthenticated: (uuid: string, token?: string) => void;
	setToken: (token: string) => void;
	logout: () => void;
};

const useAuthStore = create(
	persist<{ auth: AuthState; actions: AuthActions }>(
		(set, get) => ({
			auth: { uuid: null, token: null, loggedIn: false },
			actions: {
				setToken: (token: string) =>
					set((prev) => ({
						...prev,
						auth: {
							...prev.auth,
							token,
						},
					})),
				setAsAuthenticated: (uuid: string, token?: string) =>
					set((prev) => ({
						...prev,
						auth: {
							token: token ?? prev.auth.token,
							uuid,
							loggedIn: true,
						},
					})),
				logout: () =>
					set((prev) => ({
						...prev,
						auth: {
							token: null,
							uuid: null,
							loggedIn: false,
						},
					})),
			},
		}),
		{
			name: 'auth-storage',
			storage: createJSONStorage(() => localStorage),
			partialize: ({ actions, ...rest }: any) => rest,
		}
	)
);

export const useAuthStoreState = () => useAuthStore().auth;
export const useAuthStoreActions = () => useAuthStore().actions;
export const useAuthBearerToken = () => {
	const { auth } = useAuthStore();
	return auth.token ? `Bearer ${auth.token}` : '';
};
