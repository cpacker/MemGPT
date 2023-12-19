import { getDetailContentFromFile } from "@/app/client/fetch/file";
import { URLDetail, URLDetailContent, isURL } from "@/app/client/fetch/url";
import { Button } from "@/app/components/ui/button";
import { Textarea } from "@/app/components/ui/textarea";
import { useToast } from "@/app/components/ui/use-toast";
import { useSubmitHandler } from "@/app/hooks/useSubmit";
import { cn } from "@/app/lib/utils";
import { useBotStore } from "@/app/store/bot";
import { FileWrap } from "@/app/utils/file";
import { Send } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useDebouncedCallback } from "use-debounce";
import { ChatControllerPool } from "../../client/controller";
import {
  ALLOWED_DOCUMENT_EXTENSIONS,
  ALLOWED_IMAGE_EXTENSIONS,
  ALLOWED_TEXT_EXTENSIONS,
  DOCUMENT_FILE_SIZE_LIMIT,
} from "../../constant";
import Locale from "../../locales";
import { callSession } from "../../store";
import { autoGrowTextArea } from "../../utils/autogrow";
import { useMobileScreen } from "../../utils/mobile";
import FileUploader from "../ui/file-uploader";
import ImagePreview from "../ui/image-preview";
import { isVisionModel } from "../../client/platforms/llm";

export interface ChatInputProps {
  inputRef: React.RefObject<HTMLTextAreaElement>;
  userInput: string;
  temporaryURLInput: string;
  setUserInput: (input: string) => void;
  setTemporaryURLInput: (url: string) => void;
  scrollToBottom: () => void;
  setAutoScroll: (autoScroll: boolean) => void;
}

export default function ChatInput(props: ChatInputProps) {
  const {
    inputRef,
    userInput,
    setUserInput,
    setTemporaryURLInput,
    scrollToBottom,
    setAutoScroll,
  } = props;

  const { toast } = useToast();
  const { shouldSubmit } = useSubmitHandler();
  const isMobileScreen = useMobileScreen();

  const botStore = useBotStore();
  const bot = botStore.currentBot();
  const session = botStore.currentSession();

  const [imageFile, setImageFile] = useState<URLDetail>();
  const [temporaryBlobUrl, setTemporaryBlobUrl] = useState<string>();

  // auto grow input
  const [inputRows, setInputRows] = useState(2);
  const measure = useDebouncedCallback(
    () => {
      const rows = inputRef.current ? autoGrowTextArea(inputRef.current) : 1;
      const inputRows = Math.min(
        20,
        Math.max(1 + Number(!isMobileScreen), rows),
      );
      setInputRows(inputRows);
    },
    100,
    {
      leading: true,
      trailing: true,
    },
  );

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(measure, [userInput]);

  const onInput = (text: string) => {
    setUserInput(text);
  };

  const showError = (errMsg: string) => {
    toast({
      title: errMsg,
      variant: "destructive",
    });
  };

  const callLLM = async ({
    input,
    fileDetail,
  }: {
    input?: string;
    fileDetail?: URLDetailContent;
  }) => {
    await callSession(
      bot,
      session,
      {
        onUpdateMessages: (messages) => {
          botStore.updateBotSession((session) => {
            // trigger re-render of messages
            session.messages = messages;
          }, bot.id);
        },
      },
      input,
      fileDetail,
    );
    setImageFile(undefined);
    setTemporaryURLInput("");
    setUserInput("");
  };

  const manageTemporaryBlobUrl = (
    file: File,
    action: () => Promise<void>,
  ): Promise<void> => {
    let tempUrl: string;
    if (file.type === "image/jpeg") {
      tempUrl = URL.createObjectURL(file);
      setTemporaryBlobUrl(tempUrl);
    }

    return action().finally(() => {
      if (file.type === "image/jpeg") {
        URL.revokeObjectURL(tempUrl);
        setTemporaryBlobUrl(undefined);
      }
    });
  };

  const doSubmitFile = async (fileInput: FileWrap) => {
    try {
      await manageTemporaryBlobUrl(fileInput.file, async () => {
        const fileDetail = await getDetailContentFromFile(fileInput);
        if (fileInput.file.type === "image/jpeg") {
          setImageFile(fileDetail);
        } else {
          callLLM({ fileDetail });
        }
      });
    } catch (error) {
      showError(Locale.Upload.Failed((error as Error).message));
    }
  };

  const doSubmit = async (input: string) => {
    if (input.trim() === "") return;
    if (isURL(input)) {
      setTemporaryURLInput(input);
    }
    await callLLM({ input, fileDetail: imageFile });
    if (!isMobileScreen) inputRef.current?.focus();
    setAutoScroll(true);
  };

  // check if should send message
  const onInputKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (shouldSubmit(e)) {
      if (!isRunning && !isUploadingImage) {
        doSubmit(userInput);
      }
      e.preventDefault();
    }
  };

  const autoFocus = !isMobileScreen; // wont auto focus on mobile screen

  const isRunning = ChatControllerPool.isRunning(bot.id);

  const removeImage = () => {
    setImageFile(undefined);
  };

  const previewImage = temporaryBlobUrl || imageFile?.url;
  const isUploadingImage = temporaryBlobUrl !== undefined;

  const checkExtension = (extension: string) => {
    if (!ALLOWED_DOCUMENT_EXTENSIONS.includes(extension)) {
      return Locale.Upload.Invalid(ALLOWED_DOCUMENT_EXTENSIONS.join(","));
    }
    if (
      !isVisionModel(bot.modelConfig.model) &&
      ALLOWED_IMAGE_EXTENSIONS.includes(extension)
    ) {
      return Locale.Upload.ModelDoesNotSupportImages(
        ALLOWED_TEXT_EXTENSIONS.join(","),
      );
    }
    return null;
  };

  return (
    <div className="flex flex-1 items-end relative">
      {previewImage && (
        <div className="absolute top-[12px] left-[12px] w-[50px] h-[50px] rounded-xl cursor-pointer">
          <ImagePreview
            url={previewImage}
            uploading={isUploadingImage}
            onRemove={removeImage}
          />
        </div>
      )}
      <Textarea
        className={cn(
          "ring-inset focus-visible:ring-offset-0 pr-28 md:pr-40 min-h-[56px]",
          {
            "pt-20": previewImage,
          },
        )}
        ref={inputRef}
        placeholder={
          isMobileScreen ? Locale.Chat.InputMobile : Locale.Chat.Input
        }
        onInput={(e) => onInput(e.currentTarget.value)}
        value={userInput}
        onKeyDown={onInputKeyDown}
        onFocus={scrollToBottom}
        onClick={scrollToBottom}
        rows={inputRows}
        autoFocus={autoFocus}
      />
      <div className="my-2 flex items-center gap-2.5 absolute right-[15px]">
        <FileUploader
          config={{
            inputId: "document-uploader",
            allowedExtensions: ALLOWED_DOCUMENT_EXTENSIONS,
            checkExtension,
            fileSizeLimit: DOCUMENT_FILE_SIZE_LIMIT,
            disabled: isRunning || isUploadingImage,
          }}
          onUpload={doSubmitFile}
          onError={showError}
        />
        {isMobileScreen ? (
          <Button
            size="icon"
            onClick={() => doSubmit(userInput)}
            disabled={isRunning || isUploadingImage}
          >
            <Send className="h-4 w-4" />
          </Button>
        ) : (
          <Button
            onClick={() => doSubmit(userInput)}
            disabled={isRunning || isUploadingImage}
          >
            <Send className="h-4 w-4 mr-2" />
            {Locale.Chat.Send}
          </Button>
        )}
      </div>
    </div>
  );
}
