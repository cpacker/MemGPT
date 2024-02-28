import { parseISO } from 'date-fns';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { Message } from './message';

const reviver = (key: string, value: unknown) => {
	return key === 'date' ? parseISO(value as string) : value;
};

export type MessageHistory = {
	[key: string]: Message[];
};

const useMessageHistoryStore = create(
	persist<{ history: MessageHistory; actions: { addMessage: (key: string, message: Message) => void } }>(
		(set) => ({
			history: {},
			actions: {
				addMessage: (key: string, message: Message) =>
					set((prev) => ({
						...prev,
						history: {
							...prev.history,
							[key]: [...(prev.history[key] ?? []), message],
						},
					})),
			},
		}),
		{
			name: 'message-history-storage',
			storage: createJSONStorage(() => localStorage, { reviver }),
			partialize: ({ actions, ...rest }: any) => rest,
		}
	)
);

export const useMessageHistory = () => useMessageHistoryStore((s) => s.history);
export const useMessagesForKey = (key: string) => useMessageHistoryStore((s) => s.history[key] ?? []);

export const useMessageHistoryActions = () => useMessageHistoryStore((s) => s.actions);
