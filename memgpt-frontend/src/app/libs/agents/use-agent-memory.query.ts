import { useQuery } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';
import { AgentMemory } from './agent-memory';


export const useAgentMemoryQuery = (agent_id: string | null | undefined) => useQuery(
  {
    queryKey: ['agents',agent_id, 'memory'],
    queryFn: async () => await fetch(API_BASE_URL + `/agents/memory?agent_id=${agent_id}&user_id=null`).then(res => res.json()) as Promise<AgentMemory>,
    enabled: !!agent_id
  });
