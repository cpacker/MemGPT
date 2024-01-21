import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type AuthState = {
	uuid: string | null;
};
export type AuthActions = { setAsAuthenticated: (uuid: string) => void };

const useAuthStore = create(
	persist<{ auth: AuthState; actions: AuthActions }>(
		(set, get) => ({
			auth: { uuid: null },
			actions: {
				setAsAuthenticated: (uuid: string) =>
					set((prev) => ({
						...prev,
						auth: {
							uuid,
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
