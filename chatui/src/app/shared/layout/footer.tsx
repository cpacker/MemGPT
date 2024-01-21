import { cnH4, cnMuted } from '@memgpt/components/typography';

function Footer() {
	const year = new Date().getFullYear();
	return (
		<div className="flex items-end justify-between border-t p-8">
			<div>
				<p className={cnH4()}>MemGPT</p>
				<p className={cnMuted()}>Towards LLMs as Operating Systems</p>
			</div>
			<p className={cnMuted()}>&copy; {year} MemGPT</p>
		</div>
	);
}

export default Footer;
