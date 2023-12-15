import React from 'react';

import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { Button } from '@memgpt/components/button';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@memgpt/components/form';
import { RadioGroup, RadioGroupItem } from '@memgpt/components/radio-group';
import { toast } from '@memgpt/components/toast';
import { useAgentsQuery } from '../../../libs/agents/use-agents.query';
import { SettingsLayout } from '../layout';
import { useAgentActions, useCurrentAgent } from '../../../libs/agents/agent.store';


const agentsFormSchema = z.object({
  currentAgentName: z.string({
    required_error: 'Please select an agent.',
  }),
});

type AgentsFormValues = z.infer<typeof agentsFormSchema>

// This can come from your database or API.
const getDefaultValues: (initialName: string | null | undefined) => Partial<AgentsFormValues> = (initialName: string | null | undefined) => ({
  currentAgentName: initialName ?? '',
});

export function AgentsForm() {
  const { data, isLoading } = useAgentsQuery();
  const currentAgent = useCurrentAgent();
  const { setAgent } = useAgentActions();

  const form = useForm<AgentsFormValues>({
    resolver: zodResolver(agentsFormSchema),
    defaultValues: getDefaultValues(currentAgent?.name),
  });

  function onSubmit(formValues: AgentsFormValues) {
    const existingAgent = (data?.agents ?? []).find(agent => agent.name === formValues.currentAgentName);
    if (existingAgent) {
      setAgent(existingAgent);
      toast({
        title: 'Agent updated successfully!',
        description: 'You can now continue your conversation with them!',
      });
    }
  }

  return (
    <SettingsLayout title='Agents' description='Manage the agents you chat with...'>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className='space-y-8'>
          <FormField
            control={form.control}
            name='currentAgentName'
            render={({ field }) => (
              <FormItem className='space-y-1'>
                <FormLabel>Current Agent</FormLabel>
                <FormDescription>
                  Agent you are currently chatting with...
                </FormDescription>
                <FormMessage />
                <RadioGroup
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                  className='flex flex-wrap gap-8 pt-2'
                >
                  {data?.agents?.map((agent, i) => <FormItem key={i}>
                    <FormLabel className='[&:has([data-state=checked])>div]:border-primary'>
                      <FormControl>
                        <RadioGroupItem value={agent.name} className='sr-only' />
                      </FormControl>
                      <div className='items-center rounded-md border-2 border-muted p-1 hover:border-accent'>
                        <div className='space-y-2 rounded-sm bg-[#ecedef] p-2'>
                          <div className='space-y-2 rounded-md bg-white p-2 shadow-sm'>
                            <div className='h-2 w-[80px] rounded-lg bg-[#ecedef]' />
                            <div className='h-2 w-[100px] rounded-lg bg-[#ecedef]' />
                          </div>
                          <div className='flex items-center space-x-2 rounded-md bg-white p-2 shadow-sm'>
                            <div className='h-4 w-4 rounded-full bg-[#ecedef]' />
                            <div className='h-2 w-[100px] rounded-lg bg-[#ecedef]' />
                          </div>
                          <div className='flex items-center space-x-2 rounded-md bg-white p-2 shadow-sm'>
                            <div className='h-4 w-4 rounded-full bg-[#ecedef]' />
                            <div className='h-2 w-[100px] rounded-lg bg-[#ecedef]' />
                          </div>
                        </div>
                      </div>

                      <span className='block w-full p-2 text-center font-normal'>
                      {agent.name}
                    </span>
                    </FormLabel>
                  </FormItem>)}
                </RadioGroup>
              </FormItem>
            )}
          />

          <Button type='submit'>Update agent</Button>
        </form>
      </Form>
    </SettingsLayout>
  );
}
