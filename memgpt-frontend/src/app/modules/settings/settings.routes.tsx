import { Settings } from './settings';
import { AgentsForm } from './agents/agents';
import { RouteObject } from 'react-router-dom';
import { ProfileForm } from './profile/profile-form';

export const settingsRoute: RouteObject = {
  path: 'settings',
  element: <Settings/>,
  children: [
    {
      path: 'agents',
      element: <AgentsForm/>
    },
    {
      path: 'profile',
      element: <ProfileForm/>
    }
    ]
}
