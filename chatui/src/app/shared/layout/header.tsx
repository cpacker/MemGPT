import { Avatar, AvatarFallback, AvatarImage } from '@memgpt/components/avatar';
import { Button } from '@memgpt/components/button';
import { LucideLogOut, MoonStar, Sun } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { useAuthStoreActions } from '../../libs/auth/auth.store';
import { useTheme } from '../theme';

const twNavLink = '[&.active]:opacity-100 opacity-60';
const Header = () => {
	const { theme, toggleTheme } = useTheme();
	const { logout } = useAuthStoreActions();
	return (
		<div className="flex items-start justify-between border-b py-2 sm:px-8">
			<NavLink to="/">
				<span className="sr-only">Home</span>
				<Avatar className="border bg-white">
					<AvatarImage alt="MemGPT logo." src="/memgpt_logo_transparent.png" />
					<AvatarFallback className="border">MG</AvatarFallback>
				</Avatar>
			</NavLink>

			<nav className="flex space-x-4">
				<Button size="sm" asChild variant="link">
					<NavLink className={twNavLink} to="/">
						Home
					</NavLink>
				</Button>
				<Button size="sm" asChild variant="link">
					<NavLink className={twNavLink} to="/chat">
						Chat
					</NavLink>
				</Button>
				<Button size="sm" asChild variant="link">
					<NavLink className={twNavLink} to="/settings/agents">
						Settings
					</NavLink>
				</Button>
				<Button size="icon" variant="ghost" onClick={toggleTheme}>
					{theme === 'light' ? <MoonStar className="h-4 w-4" /> : <Sun className="w-4 w-4" />}
				</Button>
				<Button size="icon" variant="ghost" onClick={logout}>
					<LucideLogOut className="h-4 w-4" />
				</Button>
			</nav>
		</div>
	);
};
export default Header;
