import { Button } from '@memgpt/components/button';
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from '@memgpt/components/dialog';
import { Input } from '@memgpt/components/input';
import { Label } from '@memgpt/components/label';
import { Loader2 } from 'lucide-react';
import { useAgentsCreateMutation } from '../../libs/agents/use-agents.mutation';

const CreateAgentDialog = (props: { open: boolean; onOpenChange: (open: boolean) => void }) => {
	const createAgent = useAgentsCreateMutation();
	return (
		<Dialog open={props.open} onOpenChange={props.onOpenChange}>
			<DialogContent className="sm:max-w-[425px]">
				<form
					onSubmit={(event) => {
						event.preventDefault();
						const formData = new FormData(event.currentTarget);
						// ✅ mutation is invoked when the form is submitted
						createAgent.mutate(
							{
								user_id: 'placeholder',
								config: {
									name: `${formData.get('name')}`,
									human: `${formData.get('human')}`,
									persona: `${formData.get('persona')}`,
									model: `${formData.get('model')}`,
								},
							},
							{
								onSuccess: () => props.onOpenChange(false),
							}
						);
					}}
				>
					<DialogHeader>
						<DialogTitle>Create Agent</DialogTitle>
						<DialogDescription>Add a new agent here. Click create when you're done.</DialogDescription>
					</DialogHeader>
					<div className="grid gap-4 py-4">
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="name" className="text-right">
								Name
							</Label>
							<Input id="name" name="name" defaultValue="James Bond" className="col-span-3" />
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="human" className="text-right">
								Human
							</Label>
							<Input id="human" name="human" defaultValue="cs_phd" className="col-span-3" />
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="persona" className="text-right">
								Persona
							</Label>
							<Input id="persona" name="persona" defaultValue="sam_pov" className="col-span-3" />
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="model" className="text-right">
								Model
							</Label>
							<Input id="model" name="model" defaultValue="gpt-4" className="col-span-3" />
						</div>
					</div>
					<DialogFooter>
						<Button type="submit" disabled={createAgent.isPending}>
							{createAgent.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : undefined}
							{createAgent.isPending ? 'Creating Agent' : 'Create Agent'}
						</Button>
					</DialogFooter>
				</form>
			</DialogContent>
		</Dialog>
	);
};

export default CreateAgentDialog;
