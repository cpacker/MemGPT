import { BaseMessage } from './base-message';

const UserMessage = (props: { message: string; date: Date }) => {
	return (
		<BaseMessage
			message={props.message}
			date={props.date}
			dir="rtl"
			bg="bg-muted-foreground/40 dark:bg-muted-foreground/20"
			fg="text-black dark:text-white"
			initials="U"
		/>
	);
};
export default UserMessage;
