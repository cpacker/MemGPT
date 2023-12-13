import React from 'react';
import { Agent } from '../../libs/agents/agent';
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from '@memgpt/components/card';
import { Button } from '@memgpt/components/button';
import { NavLink } from 'react-router-dom';
import { Badge } from '@memgpt/components/badge';

const AgentCard = ({ name, persona, human, create_time, className, onBtnClick, isCurrentAgent }: Agent & {
  isCurrentAgent: boolean,
  className: string,
  onBtnClick: () => void
}) => <Card className={className}>
  <CardHeader>
    <CardTitle className="flex items-center justify-between">
      <span>{name}</span>
      {isCurrentAgent && <Badge variant="secondary">Current Agent</Badge>}
    </CardTitle>
    <CardDescription>{persona}</CardDescription>
  </CardHeader>
  <CardFooter>
    <Button variant="secondary" onClick={onBtnClick} asChild>
      <NavLink to="/chat">Start Chat</NavLink>
    </Button>
  </CardFooter>
</Card>;

export default AgentCard;
