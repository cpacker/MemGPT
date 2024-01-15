import { Button } from '@memgpt/components/button';
import { Skeleton } from '@memgpt/components/skeleton';
import { cnH1, cnLead } from '@memgpt/components/typography';
import { LucidePlus } from 'lucide-react';
import { useState } from 'react';
import { useAgentActions, useCurrentAgent } from '../../libs/agents/agent.store';
import { useAgentsQuery } from '../../libs/agents/use-agents.query';
import AgentCard from './agent-card';
import CreateAgentDialog from './create-agent-dialog';

const Home = () => {
	const { data, isLoading } = useAgentsQuery();
	const { setAgent } = useAgentActions();
	const currentAgent = useCurrentAgent();
	const [showingAgentCreation, setShowingAgentCreation] = useState(false);

	const agentsOrSkeletons = isLoading
		? [1, 2, 3, 4, 5, 6, 7, 8].map((_, i) => <Skeleton key={i} className="h-52 w-full flex-none opacity-30 sm:w-96" />)
		: (data?.agents ?? []).map((a) => (
				<AgentCard
					className="flex h-52 w-full flex-none flex-col justify-between shadow-md sm:w-96"
					key={a.name}
					name={a.name}
					human={a.human}
					persona={a.persona}
					created_at={a.created_at}
					onBtnClick={() => setAgent(a)}
					isCurrentAgent={!!currentAgent && currentAgent?.id === a.id}
				/>
			));

	return (
		<>
			<div className="flex h-full flex-col items-center overflow-y-scroll">
				<div className="p-2 pt-40 pb-12">
					<h1 className={cnH1()}>Welcome to MemGPT</h1>
					<p className={cnLead('mt-2 mb-4')}>Select or create an agent to start your conversation...</p>
				</div>
				<div className="mx-auto flex w-full max-w-screen-2xl flex-wrap gap-12 px-8 pb-20">
					<Button
						variant="outline"
						onClick={() => setShowingAgentCreation(true)}
						className="flex h-52 w-full flex-none flex-col items-center justify-center sm:w-96"
					>
						<LucidePlus className="h-8 w-8" />
						<span className="mt-2">Add New</span>
					</Button>
					{agentsOrSkeletons}
				</div>
			</div>
			<CreateAgentDialog open={showingAgentCreation} onOpenChange={(open) => setShowingAgentCreation(open)} />
		</>
	);
};

export default Home;
