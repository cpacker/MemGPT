import { createBrowserRouter, Outlet } from 'react-router-dom';
import { chatRoute } from './modules/chat/chat.routes';
import Home from './modules/home/home';
import { settingsRoute } from './modules/settings/settings.routes';
import Footer from './shared/layout/footer';
import Header from './shared/layout/header';

const rootRoute = () => (
	<>
		<Header />
		<div className="h-full">
			<Outlet />
		</div>
		<Footer />
	</>
);

export const router = createBrowserRouter([
	{
		path: '/',
		element: rootRoute(),
		children: [
			{
				path: '',
				element: <Home />,
			},
			chatRoute,
			settingsRoute,
		],
	},
]);
