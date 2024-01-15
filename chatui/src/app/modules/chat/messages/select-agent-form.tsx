import { Button } from '@memgpt/components/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@memgpt/components/card';
import { cn } from '@memgpt/utils';
import { LucideSmile } from 'lucide-react';
import { useState } from 'react';
import { Agent } from '../../../libs/agents/agent';
import { useAgentActions } from '../../../libs/agents/agent.store';
import { useAgentsQuery } from '../../../libs/agents/use-agents.query';

const SelectAgentForm = () => {
	const { data } = useAgentsQuery();
	const [newAgent, setNewAgent] = useState<Agent | null>(null);
	const { setAgent } = useAgentActions();
	return (
		<Card className="my-10 mx-4 w-fit bg-background animate-in slide-in-from-top slide-out-to-top duration-700 sm:mx-auto ">
			<CardHeader className="pb-3">
				<CardTitle>Choose Agent</CardTitle>
				<CardDescription>Pick an agent to start a conversation...</CardDescription>
			</CardHeader>
			<CardContent className="grid gap-1">
				{(data?.agents ?? []).map((agent, i) => (
					<button
						key={i}
						onClick={() => setNewAgent(agent)}
						className={cn(
							'-mx-2 flex items-start space-x-4 rounded-md p-2 text-left transition-all',
							newAgent?.name === agent.name
								? 'bg-accent text-accent-foreground'
								: 'hover:bg-accent hover:text-accent-foreground'
						)}
					>
						<LucideSmile className="mt-px h-5 w-5" />
						<div className="space-y-1">
							<p className="text-sm font-medium leading-none">{agent.name}</p>
							<p className="text-sm text-muted-foreground">
								{agent.human} | {agent.persona} | {agent.created_at}
							</p>
						</div>
					</button>
				))}
			</CardContent>
			<CardFooter>
				<Button onClick={() => newAgent && setAgent(newAgent)} className="w-full">
					Start Chat
				</Button>
			</CardFooter>
		</Card>
	);
};

export default SelectAgentForm;
