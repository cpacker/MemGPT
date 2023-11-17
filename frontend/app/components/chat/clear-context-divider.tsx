import { useBotStore } from "@/app/store/bot";
import Locale from "../../locales";
import { Card, CardContent } from "@/app/components/ui/card";

export function ClearContextDivider({ botId }: { botId: string }) {
  const botStore = useBotStore();

  return (
    <Card
      className="cursor-pointer hover:border-primary rounded-sm"
      onClick={() =>
        botStore.updateBotSession(
          (session) => (session.clearContextIndex = undefined),
          botId,
        )
      }
    >
      <CardContent className="p-1 group text-foreground hover:text-primary">
        <div className="text-center text-xs font-semibold">
          <span className="inline-block group-hover:hidden opacity-50">
            {Locale.Context.Clear}
          </span>
          <span className="hidden group-hover:inline-block">
            {Locale.Context.Revert}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
