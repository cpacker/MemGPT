import { Avatar, AvatarFallback, AvatarImage } from '@memgpt/components/avatar';
import { Button } from '@memgpt/components/button';
import { Input } from '@memgpt/components/input';
import { Label } from '@memgpt/components/label';
import { cnH3, cnMuted } from '@memgpt/components/typography';
import { Loader2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuthMutation } from '../../../libs/auth/use-auth.mutation';

const year = new Date().getFullYear();
const LoginPage = () => {
	const mutation = useAuthMutation();
	const navigate = useNavigate();
	return (
		<div className="relative flex h-full w-full items-center justify-center">
			<div className="-mt-40 flex max-w-sm flex-col items-center justify-center">
				<Avatar className="mb-2 h-16 w-16 border bg-white">
					<AvatarImage alt="MemGPT logo." src="/memgpt_logo_transparent.png" />
					<AvatarFallback className="border">MG</AvatarFallback>
				</Avatar>
				<h1 className={cnH3('mb-2')}>Welcome to MemGPT</h1>
				<p className="mb-6 text-muted-foreground">Sign in below to start chatting with your agent</p>
				<form
					className="w-full"
					onSubmit={(e) => {
						e.preventDefault();
						const password = new FormData(e.currentTarget).get('password') as string;
						if (!password || password.length === 0) return;
						mutation.mutate(password, {
							onSuccess: ({ uuid }, password) => setTimeout(() => navigate('/'), 600),
						});
					}}
				>
					<Label className="sr-only" htmlFor="password">
						Password
					</Label>
					<Input
						name="password"
						className="mb-2 w-full"
						type="password"
						autoComplete="off"
						autoCorrect="off"
						id="password"
					/>
					<Button type="submit" className="mb-6 w-full">
						{mutation.isPending ? (
							<span className="flex items-center animate-in slide-in-from-bottom-2">
								{/* eslint-disable-next-line react/jsx-no-undef */}
								<Loader2 className="mr-2 h-4 w-4 animate-spin" />
								Signing in
							</span>
						) : null}
						{mutation.isSuccess ? <span className="animate-in slide-in-from-bottom-2">Signed in!</span> : null}
						{!mutation.isPending && mutation.isError ? (
							<span className="animate-in slide-in-from-bottom-2">Sign In Failed. Try again...</span>
						) : null}
						{!mutation.isPending && !mutation.isSuccess && !mutation.isError ? (
							<span>Sign In with Password</span>
						) : null}
					</Button>
				</form>
				<p className="text-center text-muted-foreground">
					By clicking continue, you agree to our Terms of Service and Privacy Policy.
				</p>
			</div>
			<p className={cnMuted('absolute inset-x-0 bottom-3 text-center')}>&copy; {year} MemGPT</p>
		</div>
	);
};

export default LoginPage;
