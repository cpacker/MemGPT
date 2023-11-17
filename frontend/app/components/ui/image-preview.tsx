import { cn } from "@/app/lib/utils";
import { Loader2Icon, XCircleIcon } from "lucide-react";
import Image from "next/image";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./tooltip";

export default function ImagePreview({
  url,
  uploading,
  onRemove,
}: {
  url: string;
  uploading: boolean;
  onRemove: () => void;
}) {
  return (
    <div className="relative w-full h-full group">
      <Image
        src={url}
        alt="Uploaded image"
        fill
        className="object-cover w-full h-full rounded-xl hover:brightness-75"
      />
      <div
        className={cn(
          "absolute -top-2 -right-2 w-6 h-6 z-10 bg-gray-500 text-white rounded-full",
          { "hidden group-hover:block": !uploading },
        )}
      >
        <TooltipProvider>
          <Tooltip delayDuration={0}>
            <TooltipTrigger>
              {uploading ? (
                <Loader2Icon className="w-6 h-6 bg-gray-500 text-white rounded-full animate-spin p-1" />
              ) : (
                <XCircleIcon
                  className="w-6 h-6 bg-gray-500 text-white rounded-full"
                  onClick={onRemove}
                />
              )}
            </TooltipTrigger>
            <TooltipContent side="right">
              {uploading ? "Uploading file..." : "Remove file"}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}
