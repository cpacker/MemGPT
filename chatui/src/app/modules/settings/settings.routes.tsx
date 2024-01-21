import { RouteObject } from 'react-router-dom';
import { AgentsForm } from './agents/agents';
import { ProfileForm } from './profile/profile-form';
import { Settings } from './settings';

export const settingsRoute: RouteObject = {
	path: 'settings',
	element: <Settings />,
	children: [
		{
			path: 'agents',
			element: <AgentsForm />,
		},
		{
			path: 'profile',
			element: <ProfileForm />,
		},
	],
};
