import React, { useState } from 'react';
import { useAgentActions, useCurrentAgent } from '../../libs/agents/agent.store';
import { cnH1, cnLead } from '@memgpt/components/typography';
import { useAgentsQuery } from '../../libs/agents/use-agents.query';
import AgentCard from './agent-card';
import { LucidePlus } from 'lucide-react';
import { Skeleton } from '@memgpt/components/skeleton';
import CreateAgentDialog from './create-agent-dialog';
import { Button } from '@memgpt/components/button';

const Home = () => {
  const { data, isLoading } = useAgentsQuery();
  const { setAgent } = useAgentActions();
  const currentAgent = useCurrentAgent();
  const [showingAgentCreation, setShowingAgentCreation] = useState(false);

  const agentsOrSkeletons = isLoading ?
    [1,2,3,4,5,6,7,8].map((_,i) => <Skeleton key={i} className="h-52 flex-none opacity-30 w-full sm:w-96" />)
    : (data?.agents ?? [])
      .map((a) =>
        <AgentCard className="h-52 flex flex-col justify-between flex-none w-full sm:w-96 shadow-md" key={a.name}
                   name={a.name} human={a.human} persona={a.persona}
                   create_time={a.create_time}
                   onBtnClick={() => setAgent(a)}
                   isCurrentAgent={currentAgent?.name === a.name}
        />);

  return (
    <>
      <div className="flex-col flex items-center overflow-y-scroll h-full">
        <div className="p-2 pt-40 pb-12">
          <h1 className={cnH1()}>Welcome to MemGPT</h1>
          <p className={cnLead('mt-2 mb-4')}>Select or create an agent to start your conversation...</p>
        </div>
        <div className="w-full pb-20 mx-auto max-w-screen-2xl px-8 flex gap-12 flex-wrap">
            <Button
              variant="outline"
              onClick={() => setShowingAgentCreation(true)}
              className="h-52 flex-col items-center justify-center flex flex-none w-full sm:w-96">
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
