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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@memgpt/components/select';
import { useToast } from '@memgpt/components/toast';
import { Loader2 } from 'lucide-react';
import { useEffect } from 'react';
import { Controller, useForm } from 'react-hook-form';
import { useAgentsCreateMutation } from '../../libs/agents/use-agents.mutation';
import { useAuthStoreState } from '../../libs/auth/auth.store';
import { useHumansQuery } from '../../libs/humans/use-humans.query';
import { useModelsQuery } from '../../libs/models/use-models.query';
import { usePersonasQuery } from '../../libs/personas/use-personas.query';

interface FormData {
	name: string;
	human: string;
	persona: string;
	model: string;
}

const CreateAgentDialog = ({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) => {
	const auth = useAuthStoreState();
	const createAgent = useAgentsCreateMutation(auth.uuid);
	const { toast } = useToast();

	const {
		register,
		handleSubmit,
		control,
		formState: { errors, isSubmitting, isValid },
		reset,
	} = useForm<FormData>({
		mode: 'onBlur',
	});

	useEffect(() => {
		if (!open) {
			reset(); // Resets form on dialog close
		}
	}, [open, reset]);

	// Fetch models, humans, and personas using the custom hooks
	const { data: modelsData } = useModelsQuery(auth.uuid);
	const { data: humansData } = useHumansQuery(auth.uuid);
	const { data: personasData } = usePersonasQuery(auth.uuid);

	const onSubmit = (data: FormData) => {
		if (!auth.uuid) return;

		createAgent
			.mutateAsync({
				name: data.name,
				human: data.human,
				persona: data.persona,
				model: data.model,
			})
			.then(() => {
				onOpenChange(false);
				toast({ title: 'Agent created successfully!', duration: 5000 });
			})
			.catch((error) => {
				let errorMessage = 'Error creating agent';
				const unknownErrorMessage = `${errorMessage}: Unspecified error.`;

				try {
					const errorData = JSON.parse(error.message);
					if (errorData.detail) {
						// TODO: Modify this once the backend is changed: Backend should not return 'None' as the detail.
						// This will eventually just be:
						// errorMessage = `${errorMessage}: ${errorData.detail}`;
						if (errorData.detail == 'None') {
							errorMessage = unknownErrorMessage;
						} else {
							errorMessage = `${errorMessage}: ${errorData.detail}`;
						}
					}
				} catch (parseError) {
					errorMessage = unknownErrorMessage;
				}

				toast({
					title: errorMessage,
					duration: 5000,
				});
			});
	};

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="sm:max-w-[425px]">
				<form onSubmit={handleSubmit(onSubmit)}>
					<DialogHeader>
						<DialogTitle>Create Agent</DialogTitle>
						<DialogDescription>
							Add a new agent here. Click create when you're done. Human, Persona, and Model can be left blank to use
							default values.
						</DialogDescription>
					</DialogHeader>
					<div className="grid gap-4 py-4">
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="name" className="text-right">
								Name
							</Label>
							<Input
								id="name"
								placeholder="Enter a name"
								{...register('name', { required: true })}
								className={`col-span-3 ${errors.name ? 'border-red-500' : ''} border`}
							/>
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="human" className="text-right">
								Human
							</Label>
							<Controller
								name="human"
								control={control}
								render={({ field: { onChange, value } }) => (
									<Select value={value} onValueChange={(val) => onChange(val)}>
										<SelectTrigger className="col-span-3">
											<SelectValue placeholder="Select a human">{value}</SelectValue>
										</SelectTrigger>
										<SelectContent>
											{humansData?.humans.map((human) => (
												<SelectItem key={human.name} value={human.name}>
													{human.name}
												</SelectItem>
											))}
										</SelectContent>
									</Select>
								)}
							/>
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="persona" className="text-right">
								Persona
							</Label>
							<Controller
								name="persona"
								control={control}
								render={({ field: { onChange, value } }) => (
									<Select value={value} onValueChange={(val) => onChange(val)}>
										<SelectTrigger className="col-span-3">
											<SelectValue placeholder="Select a persona">{value}</SelectValue>
										</SelectTrigger>
										<SelectContent>
											{personasData?.personas.map((persona) => (
												<SelectItem key={persona.name} value={persona.name}>
													{persona.name}
												</SelectItem>
											))}
										</SelectContent>
									</Select>
								)}
							/>
						</div>
						<div className="grid grid-cols-4 items-center gap-4">
							<Label htmlFor="model" className="text-right">
								Model
							</Label>
							<Controller
								name="model"
								control={control}
								render={({ field: { onChange, value } }) => (
									<Select value={value} onValueChange={(val) => onChange(val)}>
										<SelectTrigger className="col-span-3">
											<SelectValue placeholder="Select a model">{value}</SelectValue>
										</SelectTrigger>
										<SelectContent>
											{modelsData?.models.map((model) => (
												<SelectItem key={model.name} value={model.name}>
													{model.name}
												</SelectItem>
											))}
										</SelectContent>
									</Select>
								)}
							/>
						</div>
					</div>
					<DialogFooter>
						<Button type="submit" disabled={isSubmitting || !isValid}>
							{isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : 'Create Agent'}
						</Button>
					</DialogFooter>
				</form>
			</DialogContent>
		</Dialog>
	);
};

export default CreateAgentDialog;
