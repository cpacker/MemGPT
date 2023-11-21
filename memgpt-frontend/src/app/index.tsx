import React from 'react';
import { Button } from '@memgpt/components/button';
import { useAgentActions } from './libs/agents/agent.store';
import { cnH1, cnLead } from '@memgpt/components/typography';
import { NavLink } from 'react-router-dom';

const Index = () => {
  const { setAgent } = useAgentActions();

  return (
    <div className='h-full flex-col flex items-center'>
      <div className='p-2 py-40'>
        <h1 className={cnH1()}>Welcome to MemGPT</h1>
        <p className={cnLead('mt-2 mb-4')}>Connect to an agent to start your conversation...</p>

        <Button asChild>
          <NavLink to='/chat'>Start chatting! </NavLink>
        </Button>
      </div>
    </div>
  );
};

export default Index;
