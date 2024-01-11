import { Separator } from '@memgpt/components/separator';
import { PropsWithChildren } from 'react';

export const SettingsLayout = ({
	children,
	title,
	description,
}: PropsWithChildren<{ title: string; description: string }>) => (
	<div className="space-y-6">
		<div>
			<h3 className="text-lg font-medium">{title}</h3>
			<p className="text-sm text-muted-foreground">{description}</p>
		</div>
		<Separator />
		{children}
	</div>
);
