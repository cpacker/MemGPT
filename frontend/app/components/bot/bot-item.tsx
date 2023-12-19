import { cn } from "@/app/lib/utils";
import { Bot } from "../../store/bot";
import BotOptions from "./bot-options";
import { BotItemContextProvider, useBot } from "./use-bot";
import { BotAvatar } from "@/app/components/ui/emoji";

function BotItemUI() {
  const { bot, isActive, ensureSession } = useBot();
  return (
    <div
      className={cn(
        "flex items-center cursor-pointer mb-2 last:mb-0 rounded-md border-2 border-muted bg-popover hover:bg-accent hover:text-accent-foreground relative",
        isActive && "border-primary",
      )}
    >
      <div
        className="flex items-center space-x-2 w-full p-4 pr-12"
        onClick={ensureSession}
      >
        <div className="w-[18px] h-[18px]">
          <BotAvatar avatar={bot.avatar} />
        </div>
        <div className="font-medium">{bot.name}</div>
      </div>
      <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
        <BotOptions />
      </div>
    </div>
  );
}

export default function BotItem(props: { bot: Bot }) {
  return (
    <BotItemContextProvider bot={props.bot}>
      <BotItemUI />
    </BotItemContextProvider>
  );
}
