import React from 'react';
import { Button } from '@memgpt/components/button';
import { Brain } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@memgpt/components/dialog';
import { useCurrentAgent } from '../../../libs/agents/agent.store';
import { useAgentMemoryQuery } from '../../../libs/agents/use-agent-memory.query';
import { MemoryForm } from './memory-form';

const MemoryView = ({ open, onOpenChange }: {open: boolean, onOpenChange: (open: boolean) => void}) => {
const currentAgent = useCurrentAgent();
const {data, isLoading} = useAgentMemoryQuery(currentAgent?.name);
  return <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent className="sm:max-w-2xl">
      <DialogHeader>
        <DialogTitle>Edit Memory</DialogTitle>
        <DialogDescription>
          This is your agents current memory. Make changes and click save to edit it.
        </DialogDescription>
      </DialogHeader>
      {isLoading || !currentAgent || !data ? <p>Loading memory</p> :
       <MemoryForm data={data} agentId={currentAgent.name} className="max-h-[80vh] overflow-auto px-1 py-4"/>}
    </DialogContent>
  </Dialog>;
};

export default MemoryView;
