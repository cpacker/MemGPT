import { Home } from "@/app/components/home";
import { Bot } from "@/app/store/bot";
import { Analytics } from "@vercel/analytics/react";
import { kv } from "@vercel/kv";

export default async function App({ params }: { params: { botId: string } }) {
  console.log(`[Share] try loading bot with key ${params.botId}`);
  let bot: Bot | null = null;
  try {
    const res: { bot: Bot } | null = await kv.get(params.botId);
    bot = res?.bot || null;
  } catch (e) {
    console.error(`[Share] failed to load bot with key ${params.botId}`, e);
  }

  if (!bot) {
    console.log(`[Share] requested unknown bot with id ${params.botId}`);
    return (
      <>
        Sorry, there is no bot at this URL. Try&nbsp;
        <a href="/">creating your own bot</a>.
      </>
    );
  }

  console.debug("[Share] bot loaded", bot);

  return (
    <>
      <Home bot={bot} />
      <Analytics />
    </>
  );
}
