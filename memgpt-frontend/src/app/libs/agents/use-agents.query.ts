import { useQuery } from '@tanstack/react-query';
import { Agent } from './agent';

const API_URL = 'http://localhost:8000/api';

export const useAgentsQuery = () => useQuery(
  {
    queryKey: ['agents'], queryFn:
      async () =>
        await fetch(API_URL + '/agents').then(res => res.json()) as Promise<Agent[]>,
  });
