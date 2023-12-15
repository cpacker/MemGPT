import React, { useState } from 'react';
import { useAgentActions, useCurrentAgent } from '../../libs/agents/agent.store';
import { cnH1, cnLead } from '@memgpt/components/typography';
import { useAgentsQuery } from '../../libs/agents/use-agents.query';
import AgentCard from './agent-card';
import { LucidePlus } from 'lucide-react';
import { Skeleton } from '@memgpt/components/skeleton';
import CreateAgentDialog from './create-agent-dialog';

const Home = () => {
  const { data, isLoading } = useAgentsQuery();
  const { setAgent } = useAgentActions();
  const currentAgent = useCurrentAgent();
  const [showingAgentCreation, setShowingAgentCreation] = useState(false);

  const agentsOrSkeletons = isLoading ? <>
      <Skeleton className="flex-none opacity-30 w-80" />
      <Skeleton className="flex-none opacity-30 w-80" />
    </>
    : (data?.agents ?? [])
      .map((a) =>
        <AgentCard className="flex-none w-80 snap-center shadow-md snap-always" key={a.name}
                   name={a.name} human={a.human} persona={a.persona}
                   create_time={a.create_time}
                   onBtnClick={() => setAgent(a)}
                   isCurrentAgent={currentAgent?.name === a.name}
        />);

  return (
    <>
      <div className="h-full flex-col flex items-center">
        <div className="p-2 pt-40 pb-12">
          <h1 className={cnH1()}>Welcome to MemGPT</h1>
          <p className={cnLead('mt-2 mb-4')}>Select or create an agent to start your conversation...</p>
        </div>
        <div className="w-full mx-auto max-w-screen-2xl px-8">
          <div
            className="no-scrollbar min-h-[200px] snap-x snap-mandatory flex gap-12 overflow-auto px-8 py-4">
            <div className="flex-none w-80"></div>
            {agentsOrSkeletons}
            <button
              onClick={() => setShowingAgentCreation(true)}
              className="text-muted-foreground flex-col items-center justify-center flex flex-none border snap-center snap-always rounded-md w-80">
              <LucidePlus className="h-8 w-8" />
              <span className="mt-2">Add New</span>
            </button>
            <div className="flex-none w-80"></div>
          </div>
        </div>
      </div>
      <CreateAgentDialog open={showingAgentCreation} onOpenChange={(open) => setShowingAgentCreation(open)} />
    </>

  );
};

export default Home;
