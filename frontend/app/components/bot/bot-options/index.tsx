import {
  ClipboardEdit,
  Copy,
  MoreHorizontal,
  Share2,
  XCircle,
} from "lucide-react";
import { useState } from "react";
import Locale from "../../../locales";
import { AlertDialog, AlertDialogTrigger } from "../../ui/alert-dialog";
import { Button } from "../../ui/button";
import { Dialog, DialogTrigger } from "../../ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../../ui/dropdown-menu";
import { useBot } from "../use-bot";
import DeleteBotDialogContent from "./delete-bot-dialog";
import EditBotDialogContent from "./edit-bot-dialog";
import ShareBotDialogContent from "./share-bot-dialog";

export default function BotOptions() {
  const { isReadOnly, isShareble, cloneBot } = useBot();
  const [dialogContent, setDialogContent] = useState<JSX.Element | null>(null);

  return (
    <Dialog>
      <AlertDialog>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <span className="sr-only">Open menu</span>
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuLabel>Options</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={cloneBot}>
              <Copy className="mr-2 w-4 h-4" />
              <span>{Locale.Bot.EditModal.Clone}</span>
            </DropdownMenuItem>
            <DialogTrigger asChild>
              <DropdownMenuItem
                disabled={isReadOnly}
                onClick={() => setDialogContent(<EditBotDialogContent />)}
              >
                <ClipboardEdit className="mr-2 w-4 h-4" />
                <span>{Locale.Bot.Item.Edit}</span>
              </DropdownMenuItem>
            </DialogTrigger>
            <AlertDialogTrigger className="w-full">
              <DropdownMenuItem
                disabled={isReadOnly && !isShareble}
                onClick={() => setDialogContent(<DeleteBotDialogContent />)}
              >
                <XCircle className="mr-2 w-4 h-4" />
                <span>{Locale.Bot.Item.Delete}</span>
              </DropdownMenuItem>
            </AlertDialogTrigger>
            <DialogTrigger asChild>
              <DropdownMenuItem
                disabled={isReadOnly}
                onClick={() => setDialogContent(<ShareBotDialogContent />)}
              >
                <Share2 className="mr-2 w-4 h-4" />
                <span>{Locale.Bot.Item.Share}</span>
              </DropdownMenuItem>
            </DialogTrigger>
          </DropdownMenuContent>
        </DropdownMenu>
        {dialogContent}
      </AlertDialog>
    </Dialog>
  );
}
