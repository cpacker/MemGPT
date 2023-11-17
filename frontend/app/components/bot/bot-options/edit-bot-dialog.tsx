import Locale from "../../../locales";
import { DialogContent, DialogHeader, DialogTitle } from "../../ui/dialog";
import { ScrollArea } from "../../ui/scroll-area";
import { Separator } from "../../ui/separator";
import BotSettings from "../bot-settings";

export default function EditBotDialogContent() {
  return (
    <DialogContent className="max-w-4xl">
      <DialogHeader>
        <DialogTitle>{Locale.Bot.EditModal.Title}</DialogTitle>
      </DialogHeader>
      <Separator />
      <ScrollArea className="h-[50vh] mt-4 pr-4">
        <BotSettings />
      </ScrollArea>
    </DialogContent>
  );
}
