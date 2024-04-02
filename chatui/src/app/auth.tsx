import { PropsWithChildren } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuthStoreActions, useAuthStoreState } from './libs/auth/auth.store';

const Auth = (props: PropsWithChildren) => {
	const { loggedIn } = useAuthStoreState();
	const { logout } = useAuthStoreActions();
	const location = useLocation();
	const navigate = useNavigate();
	if (!loggedIn && location.pathname !== '/login') {
		logout();
		navigate('/login');
	}
	return props.children;
};

export default Auth;
