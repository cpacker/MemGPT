import { createBrowserRouter, Outlet } from 'react-router-dom';
import Auth from './auth';
import { chatRoute } from './modules/chat/chat.routes';
import Home from './modules/home/home';
import { loginRoutes } from './modules/public/login/login.routes';
import { settingsRoute } from './modules/settings/settings.routes';
import Footer from './shared/layout/footer';
import Header from './shared/layout/header';

const RootRoute = () => {
	return (
		<Auth>
			<Header />
			<div className="h-full">
				<Outlet />
			</div>
			<Footer />
		</Auth>
	);
};
export const router = createBrowserRouter([
	{
		path: '/',
		element: RootRoute(),
		children: [
			{
				path: '',
				element: <Home />,
			},
			chatRoute,
			settingsRoute,
		],
	},
	loginRoutes,
]);
