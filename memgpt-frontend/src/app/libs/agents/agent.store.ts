import { create } from 'zustand';
import { Agent } from './agent';
import { combine } from 'zustand/middleware';

const useAgentStore = create(combine({
  currentAgent: null as Agent | null,
}, (set) => ({
  actions: {
    setAgent: (agent: Agent) => set({ currentAgent: agent }),
    removeAgent: () => set({ currentAgent: null }),
  },
})));

export const useCurrentAgent = () => useAgentStore(s => s.currentAgent);
export const useAgentActions = () => useAgentStore(s => s.actions);
