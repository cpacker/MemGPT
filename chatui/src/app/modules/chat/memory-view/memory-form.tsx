import { zodResolver } from '@hookform/resolvers/zod';
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
import { Textarea } from '@memgpt/components/textarea';
import { cnMuted } from '@memgpt/components/typography';
import { cn } from '@memgpt/utils';
import { Loader2, LucideCheckCheck } from 'lucide-react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { AgentMemory } from '../../../libs/agents/agent-memory';
import { AgentMemoryUpdateSchema } from '../../../libs/agents/agent-memory-update';
import { useAgentMemoryUpdateMutation } from '../../../libs/agents/use-agent-memory.mutation';
import { useAuthStoreState } from '../../../libs/auth/auth.store';

export function MemoryForm({ className, data, agentId }: { className?: string; data: AgentMemory; agentId: string }) {
	const auth = useAuthStoreState();
	const mutation = useAgentMemoryUpdateMutation(auth.uuid, agentId);

	const form = useForm<z.infer<typeof AgentMemoryUpdateSchema>>({
		resolver: zodResolver(AgentMemoryUpdateSchema),
		defaultValues: {
			persona: data?.core_memory?.persona,
			human: data?.core_memory?.human,
			user_id: auth.uuid ?? undefined
		},
	});

	function onSubmit(data: z.infer<typeof AgentMemoryUpdateSchema>) {
		mutation.mutate(data);
	}

	return (
		<Form {...form}>
			<form onSubmit={form.handleSubmit(onSubmit)} className={cn('flex flex-col gap-8', className)}>
				<FormField
					control={form.control}
					name="persona"
					render={({ field }) => (
						<FormItem>
							<FormLabel>Persona</FormLabel>
							<FormControl>
								<Textarea className="min-h-[20rem] resize-none" {...field} />
							</FormControl>
							<FormDescription>
								This is the agents core memory. It is immediately available without querying any other resources.
							</FormDescription>
							<FormMessage />
						</FormItem>
					)}
				/>
				<FormField
					control={form.control}
					name="human"
					render={({ field }) => (
						<FormItem>
							<FormLabel>Human</FormLabel>
							<FormControl>
								<Textarea className="min-h-[20rem] resize-none" {...field} />
							</FormControl>
							<FormDescription>This is what the agent knows about you so far!</FormDescription>
							<FormMessage />
						</FormItem>
					)}
				/>
				<div className="mt-4 flex items-center justify-end">
					{mutation.isPending && (
						<span className={cnMuted('mr-6 flex items-center animate-in slide-in-from-bottom')}>
							<Loader2 className="mr-2 h-4 w-4 animate-spin" />
							Saving Memory...
						</span>
					)}
					{mutation.isSuccess && (
						<span className={cnMuted('mr-6 flex items-center text-emerald-600 animate-in slide-in-from-bottom')}>
							<LucideCheckCheck className="mr-2 h-4 w-4" />
							New Memory Saved
						</span>
					)}
					<Button type="submit" disabled={mutation.isPending}>
						Save Memory
					</Button>
				</div>
			</form>
		</Form>
	);
}
