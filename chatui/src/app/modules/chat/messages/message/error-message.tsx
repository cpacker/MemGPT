import { Alert, AlertDescription, AlertTitle } from '@memgpt/components/alert';
import { AlertCircle } from 'lucide-react';

const ErrorMessage = (props: { message: string; date: Date }) => {
	return (
		<Alert className="w-fit max-w-md p-2 text-xs [&>svg]:left-2.5 [&>svg]:top-2.5" variant="destructive">
			<AlertCircle className="h-4 w-4" />
			<AlertTitle>Something went wrong...</AlertTitle>
			<AlertDescription className="text-xs">{props.message}</AlertDescription>
		</Alert>
	);
};
export default ErrorMessage;
