import { Route } from '@tanstack/react-router';
import { rootRoute } from '../../router';
import { Settings } from './settings';
import { AgentsForm } from './agents/agents';
import { ProfileForm } from './profile/profile-form';

export const settingsRoute = new Route({ getParentRoute: () => rootRoute, path: '/settings',  component: Settings })
export const agentsRoute = new Route({ getParentRoute: () => settingsRoute, path: '/agents', component: AgentsForm })
export const profileRoute = new Route({ getParentRoute: () => settingsRoute, path: '/profile', component: ProfileForm })

settingsRoute.addChildren([agentsRoute, profileRoute])
