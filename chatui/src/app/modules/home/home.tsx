import { Button } from '@memgpt/components/button';
import { Input } from '@memgpt/components/input';
import { Skeleton } from '@memgpt/components/skeleton';
import { cnH1, cnH3, cnLead, cnMuted } from '@memgpt/components/typography';
import { LucidePlus, LucideSearch } from 'lucide-react';
import { useRef, useState } from 'react';
import { useDebounce } from 'use-debounce';
import { useAgentActions, useCurrentAgent } from '../../libs/agents/agent.store';
import { useAgentsQuery } from '../../libs/agents/use-agents.query';
import { useAuthStoreState } from '../../libs/auth/auth.store';
import AgentCard from './agent-card';
import CreateAgentDialog from './create-agent-dialog';

const Home = () => {
	const { uuid } = useAuthStoreState();
	const { data, isSuccess, isLoading } = useAgentsQuery(uuid);
	const { setAgent } = useAgentActions();
	const currentAgent = useCurrentAgent();
	const [showingAgentCreation, setShowingAgentCreation] = useState(false);
	const [searchInput, setSearchInput] = useState('');
	const [debouncedInput] = useDebounce(searchInput, 300);
	const filteredAgents = (data?.agents ?? []).filter((a) => a.name.includes(debouncedInput));
	const agentsOrSkeletons = isLoading
		? [1, 2, 3, 4, 5, 6, 7, 8].map((_, i) => <Skeleton key={i} className="h-52 w-full flex-none opacity-30 sm:w-96" />)
		: filteredAgents.map((a) => (
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

	const inputRef = useRef<HTMLInputElement>(null);
	return (
		<>
			<div className="flex h-full flex-col items-center overflow-y-scroll">
				<div className="p-2 pt-40 pb-12">
					<h1 className={cnH1()}>Welcome to MemGPT</h1>
					<p className={cnLead('mt-2 mb-4')}>Select or create an agent to start your conversation...</p>
				</div>
				<div className="mx-auto mb-12 flex w-full max-w-screen-lg justify-between">
					<div className="relative">
						<Input
							value={searchInput}
							onChange={(e) => setSearchInput(e.target.value)}
							ref={inputRef}
							placeholder="Search for Agent"
							className="w-full pl-12 sm:w-80"
						/>
						<LucideSearch
							onClick={() => inputRef.current?.focus()}
							className="absolute top-1/2 left-3 z-0 h-5 w-5 -translate-y-1/2"
						/>
					</div>
					<Button onClick={() => setShowingAgentCreation(true)}>
						<LucidePlus className="h-5 w-5" />
						<span className="ml-2">Add New</span>
					</Button>
				</div>
				<div className="mx-auto flex w-full max-w-screen-2xl flex-wrap gap-12 px-8 pb-20">
					{agentsOrSkeletons}
					{isSuccess && data?.num_agents === 0 ? (
						<div className="flex w-full flex-col items-center justify-center p-20">
							<h3 className={cnH3()}>No Agents exist</h3>
							<p className={cnMuted('mt-4')}>
								Create your first agent and start chatting by clicking the Add New button.
							</p>
						</div>
					) : undefined}
				</div>
			</div>
			<CreateAgentDialog open={showingAgentCreation} onOpenChange={(open) => setShowingAgentCreation(open)} />
		</>
	);
};

export default Home;
