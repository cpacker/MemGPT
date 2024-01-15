import { Separator } from '@memgpt/components/separator';
import { Outlet } from 'react-router-dom';
import { SidebarNav } from './sidebar-nav';

const sidebarNavItems = [
	{
		title: 'Agents',
		to: './agents',
	},
];

export type SidebarNavItem = {
	title: string;
	to: string;
};

export function Settings() {
	return (
		<div className="space-y-6 p-10 pb-16">
			<div className="space-y-0.5">
				<h1 className="text-2xl font-bold tracking-tight">Settings</h1>
				<p className="text-muted-foreground">Manage your MemGPT settings, like agents, prompts, and history.</p>
			</div>
			<Separator className="my-6" />
			<div className="flex flex-col space-y-8 lg:flex-row lg:space-x-12 lg:space-y-0">
				<aside className="-mx-4 lg:w-1/5">
					<SidebarNav items={sidebarNavItems} />
				</aside>
				<div className="flex-1 lg:max-w-4xl">
					<Outlet />
				</div>
			</div>
		</div>
	);
}
