import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@memgpt/components/dialog';
import { useCurrentAgent } from '../../../libs/agents/agent.store';
import { useAgentMemoryQuery } from '../../../libs/agents/use-agent-memory.query';
import { useAuthStoreState } from '../../../libs/auth/auth.store';
import { MemoryForm } from './memory-form';

const MemoryView = ({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) => {
	const auth = useAuthStoreState();
	const currentAgent = useCurrentAgent();
	const { data, isLoading } = useAgentMemoryQuery(auth.uuid, currentAgent?.id);
	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="sm:max-w-2xl">
				<DialogHeader>
					<DialogTitle>Edit Memory</DialogTitle>
					<DialogDescription>
						This is your agents current memory. Make changes and click save to edit it.
					</DialogDescription>
				</DialogHeader>
				{isLoading || !data || !currentAgent ? (
					<p>loading..</p>
				) : (
					<MemoryForm className="max-h-[80vh] overflow-auto px-1 py-4" data={data} agentId={currentAgent.id} />
				)}
			</DialogContent>
		</Dialog>
	);
};

export default MemoryView;
