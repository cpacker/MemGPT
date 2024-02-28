import { Button } from '@memgpt/components/button';
import { Input } from '@memgpt/components/input';
import { useState } from 'react';

import { zodResolver } from '@hookform/resolvers/zod';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@memgpt/components/form';
import { Brain } from 'lucide-react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import MemoryView from './memory-view/memory-view';

const formSchema = z.object({
	message: z.string().min(1, 'Message cannot be empty...'),
});
const UserInput = (props: { enabled: boolean; onSend: (message: string) => void }) => {
	const [open, setOpen] = useState(false);
	const form = useForm<z.infer<typeof formSchema>>({
		resolver: zodResolver(formSchema),
		defaultValues: {
			message: '',
		},
	});
	function onSubmit(values: z.infer<typeof formSchema>) {
		props.onSend(values.message);
		form.reset();
	}

	return (
		<Form {...form}>
			<form onSubmit={form.handleSubmit(onSubmit)} className="mb-8 mt-4 flex items-start justify-between gap-2">
				<FormField
					control={form.control}
					name="message"
					render={({ field }) => (
						<FormItem className="w-full">
							<FormLabel>What's on your mind</FormLabel>
							<FormControl className="w-full">
								<Input className="w-full" placeholder="Type something..." {...field} />
							</FormControl>
							<FormMessage />
						</FormItem>
					)}
				/>
				<div className="mt-8 flex gap-2">
					<Button disabled={!props.enabled} type="submit">
						Send
					</Button>
					<Button onClick={() => setOpen(true)} className="ml-1" type="button" size="icon" variant="outline">
						<Brain className="h-4 w-4" />
					</Button>
				</div>
			</form>
			<MemoryView open={open} onOpenChange={(open) => setOpen(open)} />
		</Form>
	);
};

export default UserInput;
