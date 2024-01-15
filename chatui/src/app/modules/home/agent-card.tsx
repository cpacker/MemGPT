import { Badge } from '@memgpt/components/badge';
import { Button } from '@memgpt/components/button';
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from '@memgpt/components/card';
import { NavLink } from 'react-router-dom';
import { Agent } from '../../libs/agents/agent';

const AgentCard = ({
	name,
	persona,
	human,
	created_at,
	className,
	onBtnClick,
	isCurrentAgent,
}: Omit<Agent, 'id'> & {
	isCurrentAgent: boolean;
	className: string;
	onBtnClick: () => void;
}) => (
	<Card className={className}>
		<CardHeader>
			<CardTitle className="flex items-center justify-between">
				<span>{name}</span>
				{isCurrentAgent && <Badge className="whitespace-nowrap">Current Agent</Badge>}
			</CardTitle>
			<CardDescription>{persona}</CardDescription>
		</CardHeader>
		<CardFooter>
			<Button variant="secondary" onClick={onBtnClick} asChild>
				<NavLink to="/chat">Start Chat</NavLink>
			</Button>
		</CardFooter>
	</Card>
);

export default AgentCard;
