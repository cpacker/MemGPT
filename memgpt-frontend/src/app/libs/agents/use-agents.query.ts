import { useQuery } from '@tanstack/react-query';
import { Agent } from './agent';
import { API_BASE_URL } from '../constants';

export const useAgentsQuery = () => useQuery(
  {
    queryKey: ['agents'], queryFn:
      async () =>
        await fetch(API_BASE_URL + '/agents?user_id=null').then(res => res.json()) as Promise<{num_agents: number, agents: Agent[]}>,
  });
