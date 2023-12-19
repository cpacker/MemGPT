import { useBot } from "@/app/components/bot/use-bot";
import EmojiPicker, { Theme as EmojiTheme } from "emoji-picker-react";
import { useState } from "react";
import Locale from "../../../locales";
import { Card, CardContent } from "../../ui/card";
import { Checkbox } from "../../ui/checkbox";
import { Input } from "../../ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "../../ui/popover";
import ConfigItem from "./config-item";
import { BotAvatar, getEmojiUrl } from "@/app/components/ui/emoji";

export default function BotConfig() {
  const { bot, updateBot } = useBot();
  const [showPicker, setShowPicker] = useState(false);
  return (
    <>
      <div className="font-semibold mb-2">{Locale.Bot.Config.Title}</div>
      <Card>
        <CardContent className="divide-y p-5">
          <ConfigItem title={Locale.Bot.Config.Avatar}>
            <Popover open={showPicker}>
              <PopoverTrigger onClick={() => setShowPicker(true)}>
                <BotAvatar avatar={bot.avatar} />
              </PopoverTrigger>
              <PopoverContent align="end" className="w-fit">
                <EmojiPicker
                  lazyLoadEmojis
                  theme={EmojiTheme.AUTO}
                  getEmojiUrl={getEmojiUrl}
                  onEmojiClick={(e) => {
                    updateBot((bot) => (bot.avatar = e.unified));
                    setShowPicker(false);
                  }}
                />
              </PopoverContent>
            </Popover>
          </ConfigItem>
          <ConfigItem title={Locale.Bot.Config.Name}>
            <Input
              type="text"
              value={bot.name}
              onInput={(e) =>
                updateBot((bot) => {
                  bot.name = e.currentTarget.value;
                })
              }
            />
          </ConfigItem>
          <ConfigItem
            title={Locale.Bot.Config.HideContext.Title}
            subTitle={Locale.Bot.Config.HideContext.SubTitle}
          >
            <Checkbox
              checked={bot.hideContext}
              onCheckedChange={(checked) => {
                updateBot((bot) => {
                  bot.hideContext = Boolean(checked);
                });
              }}
            />
          </ConfigItem>
          <ConfigItem
            title={Locale.Bot.Config.BotHello.Title}
            subTitle={Locale.Bot.Config.BotHello.SubTitle}
          >
            <Input
              type="text"
              value={bot.botHello || ""}
              onChange={(e) => {
                updateBot((bot) => {
                  bot.botHello = e.currentTarget.value;
                });
              }}
            />
          </ConfigItem>
        </CardContent>
      </Card>
    </>
  );
}
