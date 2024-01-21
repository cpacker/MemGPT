import { PropsWithChildren } from 'react';
import { useAuthStoreActions, useAuthStoreState } from './libs/auth/auth.store';
import { useAuthQuery } from './libs/auth/use-auth.query';

const Auth = (props: PropsWithChildren) => {
	const result = useAuthQuery();
	const { uuid } = useAuthStoreState();
	const { setAsAuthenticated } = useAuthStoreActions();
	if (result.isSuccess && uuid !== result.data.uuid) {
		setAsAuthenticated(result.data.uuid);
	}
	return props.children;
};

export default Auth;
