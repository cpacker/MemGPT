import { Button } from '@memgpt/components/button';
import { Input } from '@memgpt/components/input';
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from '@memgpt/components/dialog';
import { Select, SelectContent, SelectValue, SelectTrigger, SelectGroup, SelectLabel, SelectItem } from '@memgpt/components/select';
import { Label } from '@memgpt/components/label';
import { Loader2 } from 'lucide-react';
import { useAgentsCreateMutation } from '../../libs/agents/use-agents.mutation';
import { useAuthStoreState } from '../../libs/auth/auth.store';
import { useState, useEffect } from 'react';
import { useModelsQuery } from '../../libs/models/use-models.query';
import { usePersonasQuery } from '../../libs/personas/use-personas.query';
import { useHumansQuery } from '../../libs/humans/use-humans.query';

type DropdownValues = {
    name: string[];
    human: string[];
    persona: string[];
    model: string[];
};

const CreateAgentDialog = (props: { open: boolean; onOpenChange: (open: boolean) => void }) => {
	const auth = useAuthStoreState();
	const createAgent = useAgentsCreateMutation(auth.uuid);

    const [dropdownValues, setDropdownValues] = useState<DropdownValues>({ 
		name: [], 
		human: [], 
		persona: [], 
		model: [] 
	});

    // Fetch models, humans, and personas using the custom hooks
    const { data: modelsData } = useModelsQuery(auth.uuid);
    const { data: humansData } = useHumansQuery(auth.uuid);
    const { data: personasData } = usePersonasQuery(auth.uuid);

    // Effect to update the dropdown values when the data is fetched
    useEffect(() => {
		if (Array.isArray(modelsData?.models)) {
			setDropdownValues(prev => ({ ...prev, model: modelsData.models.map(m => m.name) }));
		}
		if (Array.isArray(humansData?.humans)) {
			setDropdownValues(prev => ({ ...prev, human: humansData.humans.map(h => h.name) }));
		}
		if (Array.isArray(personasData?.personas)) {
			setDropdownValues(prev => ({ ...prev, persona: personasData.personas.map(p => p.name) }));
		}
	}, [modelsData, humansData, personasData]);

	const [selectedValues, setSelectedValues] = useState({ name: '', human: '', persona: '', model: '' });

    const handleSelect = (fieldName: string, value: string) => {
        setSelectedValues(prev => ({ ...prev, [fieldName]: value }));
    };

	return (
		<Dialog open={props.open} onOpenChange={props.onOpenChange}>
			<DialogContent className="sm:max-w-[425px]">
				<form
					onSubmit={(event) => {
						event.preventDefault();
						if (!auth.uuid) return;
						const formData = new FormData(event.currentTarget);
						// ✅ mutation is invoked when the form is submitted
						createAgent.mutate(
							{
								name: `${formData.get('name')}`,
								human: selectedValues.human || 'cs_phd',
								persona: selectedValues.persona || 'sam_pov',
								model: selectedValues.model || 'gpt-4',
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
							<Select>
								<SelectTrigger className='col-span-3'>
									<SelectValue placeholder="Select a human" />
								</SelectTrigger>
								<SelectContent>
									<SelectGroup>
										<SelectLabel>Humans</SelectLabel>
										{dropdownValues.human.map(value => (
											<SelectItem value={value} defaultValue={'cs_phd'} key={value} onSelect={() => handleSelect('human', value)}>
												{value}
											</SelectItem>
										))}
									</SelectGroup>
								</SelectContent>
							</Select>
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="persona" className="text-right">
								Persona
							</Label>
							<Select>
								<SelectTrigger className='col-span-3'>
									<SelectValue placeholder="Select a persona" />
								</SelectTrigger>
								<SelectContent>
									<SelectGroup>
										<SelectLabel>Personas</SelectLabel>
										{dropdownValues.persona.map(value => (
											<SelectItem value={value} defaultValue={'sam_pov'} key={value} onSelect={() => handleSelect('persona', value)}>
												{value}
											</SelectItem>
										))}
									</SelectGroup>
								</SelectContent>
							</Select>
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="model" className="text-right">
								Model
							</Label>
							<Select>
								<SelectTrigger className='col-span-3'>
									<SelectValue placeholder="Select a model" />
								</SelectTrigger>
								<SelectContent>
									<SelectGroup>
										<SelectLabel>Models</SelectLabel>
										{dropdownValues.model.map(value => (
											<SelectItem value={value} defaultValue={'gpt-4'} key={value} onSelect={() => handleSelect('model', value)}>
												{value}
											</SelectItem>
										))}
									</SelectGroup>
								</SelectContent>
							</Select>
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
