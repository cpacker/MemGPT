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
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@memgpt/components/dropdown-menu'
import { Label } from '@memgpt/components/label';
import { Loader2 } from 'lucide-react';
import { useAgentsCreateMutation } from '../../libs/agents/use-agents.mutation';
import { useAuthStoreState } from '../../libs/auth/auth.store';
import { useState, useEffect } from 'react';
import { useModelsQuery } from '../../libs/models/use-models.query';
import { usePersonasQuery } from '../../libs/personas/use-personas.query';
import { useHumansQuery } from '../../libs/humans/use-humans.query';
import { useModelStore } from '../../libs/models/model.store';
import { usePersonaStore } from '../../libs/personas/persona.store';
import { useHumanStore } from '../../libs/humans/human.store';

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

    // Zustand stores to manage the state
    const { setModels } = useModelStore();
    const { setHumans } = useHumanStore();
    const { setPersonas } = usePersonaStore();

    // Effect to update the dropdown values when the data is fetched
    useEffect(() => {
		if (Array.isArray(modelsData?.models)) {
			setModels(modelsData.models);
			setDropdownValues(prev => ({ ...prev, model: modelsData.models.map(m => m.name) }));
		}
		if (Array.isArray(humansData?.humans)) {
			setHumans(humansData.humans);
			setDropdownValues(prev => ({ ...prev, human: humansData.humans.map(h => h.name) }));
		}
		if (Array.isArray(personasData?.personas)) {
			setPersonas(personasData.personas);
			setDropdownValues(prev => ({ ...prev, persona: personasData.personas.map(p => p.name) }));
		}
	}, [modelsData, humansData, personasData, setModels, setHumans, setPersonas]);

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
							<DropdownMenu>
								<DropdownMenuTrigger asChild>
									<button className='col-span-3'>{selectedValues.human || dropdownValues.human[0]}</button>
								</DropdownMenuTrigger>
								<DropdownMenuContent>
									{dropdownValues.human.map(value => (
										<DropdownMenuItem key={value} onSelect={() => handleSelect('human', value)}>
											{value}
										</DropdownMenuItem>
									))}
								</DropdownMenuContent>
							</DropdownMenu>
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="persona" className="text-right">
								Persona
							</Label>
							<DropdownMenu>
								<DropdownMenuTrigger asChild>
									<button className='col-span-3'>{selectedValues.persona || dropdownValues.persona[0]}</button>
								</DropdownMenuTrigger>
								<DropdownMenuContent>
									{dropdownValues.persona.map(value => (
										<DropdownMenuItem key={value} onSelect={() => handleSelect('persona', value)}>
											{value}
										</DropdownMenuItem>
									))}
								</DropdownMenuContent>
							</DropdownMenu>
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="model" className="text-right">
								Model
							</Label>
							<DropdownMenu>
								<DropdownMenuTrigger asChild>
									<button className='col-span-3'>{selectedValues.model || dropdownValues.model[0]}</button>
								</DropdownMenuTrigger>
								<DropdownMenuContent>
									{dropdownValues.model.map(value => (
										<DropdownMenuItem key={value} onSelect={() => handleSelect('model', value)}>
											{value}
										</DropdownMenuItem>
									))}
								</DropdownMenuContent>
							</DropdownMenu>
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
