import { RouteObject } from 'react-router-dom';
import LoginPage from './login.page';

export const loginRoutes: RouteObject = {
	path: 'login',
	element: <LoginPage />,
};
