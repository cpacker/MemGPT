import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Agent } from './agent';

const useAgentStore = create(
	persist<{
		currentAgent: Agent | null;
		lastAgentInitMessage: { date: Date; agentId: string } | null;
		actions: {
			setAgent: (agent: Agent) => void;
			setLastAgentInitMessage: (messageInfo: { date: Date; agentId: string } | null) => void;
			removeAgent: () => void;
		};
	}>(
		(set, get) => ({
			currentAgent: null,
			lastAgentInitMessage: null,
			actions: {
				setAgent: (agent: Agent) => set({ currentAgent: agent }),
				setLastAgentInitMessage: (
					messageInfo: {
						date: Date;
						agentId: string;
					} | null
				) => set((prev) => ({ ...prev, lastAgentInitMessage: messageInfo })),
				removeAgent: () => set((prev) => ({ ...prev, currentAgent: null })),
			},
		}),
		{
			name: 'agent-storage',
			partialize: ({ actions, ...rest }: any) => rest,
		}
	)
);

export const useCurrentAgent = () => useAgentStore((s) => s.currentAgent);
export const useLastAgentInitMessage = () => useAgentStore((s) => s.lastAgentInitMessage);
export const useAgentActions = () => useAgentStore((s) => s.actions);
