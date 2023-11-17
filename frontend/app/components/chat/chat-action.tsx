import { Button } from "@/app/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/app/components/ui/tooltip";

export function ChatAction(props: {
  text: string;
  icon: JSX.Element;
  onClick: () => void;
  showTitle?: boolean;
  buttonVariant?: "ghost" | "outline";
}) {
  const { text, icon, onClick, showTitle, buttonVariant } = props;
  const buttonVariantDefault = "ghost";
  const variant = buttonVariant || buttonVariantDefault;

  if (!showTitle) {
    return (
      <TooltipProvider delayDuration={200}>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="px-1">
              <Button
                size="icon"
                variant={variant}
                className="group"
                onClick={onClick}
              >
                {icon}
              </Button>
            </div>
          </TooltipTrigger>
          <TooltipContent side="top" className="text-xs text-muted-foreground">
            {text}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return (
    <Button size="sm" variant={variant} className="group" onClick={onClick}>
      {icon}
      <div className="text-xs text-muted-foreground ml-2">{text}</div>
    </Button>
  );
}
