import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

const API_URL = 'http://localhost:8000/api';


export type Agent = {
  name: string
}

export const useAgentsQuery = () => useQuery(
  {
    queryKey: ['agents'], queryFn:
      async () =>
        await fetch(API_URL + '/agents').then(res => res.json()) as Promise<Agent[]>,
  });
