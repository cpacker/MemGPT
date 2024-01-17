import { useQuery } from '@tanstack/react-query';
import { API_BASE_URL } from '../constants';

export const useMessagesQuery = (
	userId: string | null | undefined,
	agentId: string | null | undefined,
	start = 0,
	count = 10
) =>
	useQuery({
		queryKey: [userId, 'agents', 'item', agentId, 'messages', 'list', start, count],
		queryFn: async () =>
			(await fetch(
				API_BASE_URL + `/agents/message?agent_id=${agentId}&user_id=${userId}&start=${start}&count=${count}`
			).then((res) => res.json())) as Promise<{
				messages: { role: string; name: string; content: string }[];
			}>,
		enabled: !!userId && !!agentId,
	});
