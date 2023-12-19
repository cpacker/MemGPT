import { buttonVariants } from "@/app/components/ui/button";
import { cn } from "@/app/lib/utils";
import { FileWrap } from "@/app/utils/file";
import { ChangeEvent, useState } from "react";
import Locale from "../../locales";
import { Paperclip, Loader2 } from "lucide-react";

export interface FileUploaderProps {
  config?: {
    inputId?: string;
    fileSizeLimit?: number;
    allowedExtensions?: string[];
    checkExtension?: (extension: string) => string | null;
    disabled: boolean;
  };
  onUpload: (file: FileWrap) => Promise<void>;
  onError: (errMsg: string) => void;
}

const DEFAULT_INPUT_ID = "fileInput";
const DEFAULT_FILE_SIZE_LIMIT = 1024 * 1024 * 50; // 50 MB

export default function FileUploader({
  config,
  onUpload,
  onError,
}: FileUploaderProps) {
  const [uploading, setUploading] = useState(false);

  const inputId = config?.inputId || DEFAULT_INPUT_ID;
  const fileSizeLimit = config?.fileSizeLimit || DEFAULT_FILE_SIZE_LIMIT;
  const allowedExtensions = config?.allowedExtensions;
  const defaultCheckExtension = (extension: string) => {
    if (allowedExtensions && !allowedExtensions.includes(extension)) {
      return Locale.Upload.Invalid(allowedExtensions!.join(","));
    }
    return null;
  };
  const checkExtension = config?.checkExtension ?? defaultCheckExtension;

  const isFileSizeExceeded = (file: FileWrap) => {
    return file.size > fileSizeLimit;
  };

  const resetInput = () => {
    const fileInput = document.getElementById(inputId) as HTMLInputElement;
    fileInput.value = "";
  };

  const onFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const fileWrap = new FileWrap(file);
    await handleUpload(fileWrap);
    resetInput();
    setUploading(false);
  };

  const handleUpload = async (file: FileWrap) => {
    const extensionError = checkExtension(file.extension);
    if (extensionError) {
      return onError(extensionError);
    }

    if (isFileSizeExceeded(file)) {
      return onError(Locale.Upload.SizeExceeded(fileSizeLimit / 1024 / 1024));
    }

    await onUpload(file);
  };

  return (
    <div className="self-stretch">
      <input
        type="file"
        id={inputId}
        style={{ display: "none" }}
        onChange={onFileChange}
        accept={allowedExtensions?.join(",")}
        disabled={config?.disabled || uploading}
      />
      <label
        htmlFor={inputId}
        className={cn(
          buttonVariants({ variant: "secondary", size: "icon" }),
          "cursor-pointer",
          uploading && "opacity-50",
        )}
      >
        {uploading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Paperclip className="-rotate-45 w-4 h-4" />
        )}
      </label>
    </div>
  );
}
