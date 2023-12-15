import { create } from 'zustand';
import { Message } from './message';
import { combine } from 'zustand/middleware';

export type MessageHistory = {
  [key: string]: Message[]
}

const useMessageHistoryStore = create(
  combine({
      history: {} as MessageHistory,
    }, (set) => ({
      actions: {
        addMessage: (key: string, message: Message) => set(prev => ({
          ...prev,
          history: {
            ...prev.history,
            [key]: [...(prev.history[key] ?? []), message],
          },
        })),
      },
    }),
  ),
);

export const useMessageHistory = () => useMessageHistoryStore(s => s.history);
export const useMessagesForKey = (key: string) => useMessageHistoryStore(s => s.history[key] ?? []);

export const useMessageHistoryActions = () =>
  useMessageHistoryStore((s) => s.actions);
