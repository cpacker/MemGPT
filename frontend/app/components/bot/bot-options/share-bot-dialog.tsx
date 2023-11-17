import { ShareResponse } from "@/app/api/share/route";
import { Card, CardContent } from "@/app/components/ui/card";
import { Input } from "@/app/components/ui/input";
import { Loading } from "@/app/components/ui/loading";
import { useToast } from "@/app/components/ui/use-toast";
import { Bot } from "@/app/store/bot";
import { copyToClipboard } from "@/app/utils/clipboard";
import { Copy } from "lucide-react";
import { useEffect } from "react";
import { useMutation } from "react-query";
import Locale from "../../../locales";
import { Button } from "../../ui/button";
import {
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../ui/dialog";
import { useBot } from "../use-bot";

async function share(bot: Bot): Promise<ShareResponse> {
  const res = await fetch("/api/share", {
    method: "POST",
    body: JSON.stringify({ bot: bot }),
  });
  const json = await res.json();
  console.log("[Share]", json);
  if (!res.ok) {
    throw new Error(json.msg);
  }
  return json;
}

export default function ShareBotDialogContent() {
  const { toast } = useToast();
  const { bot, updateBot } = useBot();

  const shareMutation = useMutation(share, {
    onSuccess: (data) => {
      updateBot((bot) => {
        bot.share = { ...bot.share, id: data.key };
      });
    },
  });

  // FIXME: check dependency warning
  useEffect(() => {
    shareMutation.mutate(bot);
  }, []);

  return (
    <DialogContent className="max-w-3xl">
      <DialogHeader>
        <DialogTitle>{Locale.Share.Title}</DialogTitle>
      </DialogHeader>
      <div>
        {!shareMutation.error && (
          <Card>
            <CardContent className="divide-y pt-6">
              <div className="flex items-center px-3 py-0 gap-4 text-sm">
                <div className="font-medium">{Locale.Share.Url.Title}</div>
                {shareMutation.data ? (
                  <div className="flex flex-1 gap-4">
                    <Input
                      className="flex-1 w-full"
                      type="text"
                      value={shareMutation.data.url}
                      readOnly
                    />
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() =>
                        copyToClipboard(shareMutation.data.url, toast)
                      }
                    >
                      <Copy className="w-4 h-4" />
                    </Button>
                  </div>
                ) : (
                  <div className="flex flex-1 justify-end items-center text-muted-foreground gap-1">
                    <Loading />
                    Loading...
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
      <DialogFooter>
        <div>
          {shareMutation.error ? (
            <span className="text-destructive">{Locale.Share.Url.Error}</span>
          ) : (
            <div>{Locale.Share.Url.Hint}</div>
          )}
        </div>
      </DialogFooter>
    </DialogContent>
  );
}
