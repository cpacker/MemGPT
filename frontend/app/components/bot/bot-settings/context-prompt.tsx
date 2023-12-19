import { Button } from "@/app/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/app/components/ui/select";
import { Textarea } from "@/app/components/ui/textarea";
import { ArrowDownLeftSquare, PlusCircle, XCircle } from "lucide-react";
import { useQuery } from "react-query";
import { MESSAGE_ROLES } from "../../../client/platforms/llm";
import Locale from "../../../locales";
import { ChatMessage } from "../../../store";
import { fetchSiteContent, isURL } from "../../../client/fetch/url";

interface PromptInputStatusProps {
  status: "loading" | "success" | "error";
  detail: string;
}

const promptInputStatusStyle = {
  loading: "text-yellow-500",
  success: "text-primary",
  error: "text-destructive",
};

function ContextPromptInputStatus(props: PromptInputStatusProps) {
  return (
    <div className={promptInputStatusStyle[props.status]}>{props.detail}</div>
  );
}

function ContextPromptItem(props: {
  index: number;
  prompt: ChatMessage;
  update: (prompt: ChatMessage) => void;
  remove: () => void;
  insert: () => void;
}) {
  const requiredUrlInput = props.prompt.role === "URL";
  const currentInputValue = props.prompt.urlDetail
    ? props.prompt.urlDetail.url
    : props.prompt.content;
  const invalidUrlInput =
    !!currentInputValue && requiredUrlInput && !isURL(currentInputValue);
  const isFetchContentSuccess = requiredUrlInput && !!props.prompt.urlDetail;

  const { isLoading, error } = useQuery(
    ["content", currentInputValue],
    () => fetchSiteContent(currentInputValue),
    {
      enabled: requiredUrlInput && isURL(currentInputValue),
      refetchOnWindowFocus: false,
      retry: false,
      onSuccess: (urlDetail) => {
        props.update({
          ...props.prompt,
          content: urlDetail.content!,
          urlDetail,
        });
      },
    },
  );

  const handleUpdatePrompt = async (input: string) => {
    props.update({
      ...props.prompt,
      content: input,
      urlDetail: undefined,
    });
  };

  const getPromptInputStatus = (): PromptInputStatusProps | undefined => {
    if (invalidUrlInput) {
      return {
        status: "error",
        detail: "Please enter a valid URL",
      };
    }

    const errorMsg = (error as any)?.message;
    if (errorMsg) {
      return {
        status: "error",
        detail: errorMsg,
      };
    }

    if (isLoading) {
      return {
        status: "loading",
        detail: "Fetching site content...",
      };
    }

    if (isFetchContentSuccess) {
      return {
        status: "success",
        detail: "The URL has been successfully retrieved.",
      };
    }

    return undefined;
  };

  const promptInputStatus = getPromptInputStatus();

  return (
    <>
      {promptInputStatus && <ContextPromptInputStatus {...promptInputStatus} />}
      <div className="flex justify-center gap-2 w-full group items-start py-2">
        <div className="flex gap-2 items-center">
          <Select
            value={props.prompt.role}
            onValueChange={(value) =>
              props.update({
                ...props.prompt,
                role: value as any,
              })
            }
          >
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="Select role" />
            </SelectTrigger>
            <SelectContent>
              {MESSAGE_ROLES.map((r) => (
                <SelectItem key={r} value={r}>
                  {r}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Textarea
          value={currentInputValue}
          className={
            "flex-1 max-w-full text-left min-h-0 ring-inset focus-visible:ring-offset-0"
          }
          rows={4}
          onBlur={() => {
            // If the selection is not removed when the user loses focus, some
            // extensions like "Translate" will always display a floating bar
            window?.getSelection()?.removeAllRanges();
          }}
          onInput={(e) => handleUpdatePrompt(e.currentTarget.value)}
        />
        <div className="flex flex-col space-y-2">
          <Button
            variant="destructive"
            size="icon"
            onClick={() => props.remove()}
            className="h-8 w-8"
          >
            <XCircle className="w-5 h-5" />
          </Button>
          <Button
            variant="secondary"
            size="icon"
            onClick={() => props.insert()}
            className="h-8 w-8"
          >
            <ArrowDownLeftSquare className="w-5 h-5" />
          </Button>
        </div>
      </div>
    </>
  );
}

export function ContextPrompts(props: {
  context: ChatMessage[];
  updateContext: (updater: (context: ChatMessage[]) => void) => void;
}) {
  const context = props.context;

  const addContextPrompt = (prompt: ChatMessage, i: number) => {
    props.updateContext((context) => context.splice(i, 0, prompt));
  };

  const createNewEmptyPrompt = () => {
    addContextPrompt(
      {
        role: "user",
        content: "",
      },
      props.context.length,
    );
  };

  const removeContextPrompt = (i: number) => {
    props.updateContext((context) => context.splice(i, 1));
  };

  const updateContextPrompt = (i: number, prompt: ChatMessage) => {
    props.updateContext((context) => (context[i] = prompt));
  };

  return (
    <>
      <div className="mb-5">
        <div className="font-semibold mb-2 flex items-center justify-between">
          <span>{Locale.Context.Title}</span>
          <Button variant="secondary" onClick={createNewEmptyPrompt}>
            <PlusCircle className="mr-2 h-4 w-4" /> {Locale.Context.Add}
          </Button>
        </div>
        {context.map((c, i) => (
          <div key={i} className="p-2">
            <ContextPromptItem
              index={i}
              prompt={c}
              update={(prompt) => updateContextPrompt(i, prompt)}
              remove={() => removeContextPrompt(i)}
              insert={() => {
                addContextPrompt(
                  {
                    role: "user",
                    content: "",
                  },
                  i + 1,
                );
              }}
            />
          </div>
        ))}
      </div>
    </>
  );
}
