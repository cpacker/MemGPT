import { useQuery } from '@tanstack/react-query';
import { useAuthBearerToken } from '../auth/auth.store';
import { API_BASE_URL } from '../constants';

export const useMessagesQuery = (
	userId: string | null | undefined,
	agentId: string | null | undefined,
	start = 0,
	count = 10
) => {
	const bearerToken = useAuthBearerToken();
	return useQuery({
		queryKey: [userId, 'agents', 'item', agentId, 'messages', 'list', start, count],
		queryFn: async () =>
			(await fetch(
				API_BASE_URL + `/agents/${agentId}/message?user_id=${userId}&start=${start}&count=${count}`,
				{
					headers: {
						Authorization: bearerToken,
					},
				}
			).then((res) => res.json())) as Promise<{
				messages: { role: string; name: string; content: string }[];
			}>,
		enabled: !!userId && !!agentId,
	});
};
