import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/app/components/ui/hover-card";
import { Loading } from "@/app/components/ui/loading";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { useToast } from "@/app/components/ui/use-toast";
import { useScrollToBottom } from "@/app/hooks/useScroll";
import { cn } from "@/app/lib/utils";
import { useBotStore } from "@/app/store/bot";
import { copyToClipboard } from "@/app/utils/clipboard";
import { Clipboard, Eraser, PauseCircle, Trash } from "lucide-react";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import { ChatControllerPool } from "../../client/controller";
import { CHAT_PAGE_SIZE, REQUEST_TIMEOUT_MS } from "../../constant";
import Locale from "../../locales";
import { ChatMessage, createMessage } from "../../store";
import { prettyObject } from "../../utils/format";
import { useMobileScreen } from "../../utils/mobile";
import { Separator } from "../ui/separator";
import { ChatAction } from "./chat-action";
import ChatHeader from "./chat-header";
import ChatInput from "./chat-input";
import { ClearContextDivider } from "./clear-context-divider";

const Markdown = dynamic(
  async () => (await import("../ui/markdown")).Markdown,
  {
    loading: () => <Loading />,
  },
);

export function Chat() {
  const { toast } = useToast();
  const isMobileScreen = useMobileScreen();
  const botStore = useBotStore();
  const bot = botStore.currentBot();
  const session = botStore.currentSession();

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [userInput, setUserInput] = useState("");
  const [temporaryURLInput, setTemporaryURLInput] = useState("");
  const { scrollRef, setAutoScroll, scrollDomToBottom } = useScrollToBottom();

  useEffect(() => {
    botStore.updateBotSession((session) => {
      const stopTiming = Date.now() - REQUEST_TIMEOUT_MS;
      session.messages.forEach((m) => {
        // check if should stop all stale messages
        if (m.isError || (m.date && new Date(m.date).getTime() < stopTiming)) {
          if (m.streaming) {
            m.streaming = false;
          }

          if (m.content.length === 0) {
            m.isError = true;
            m.content = prettyObject({
              error: true,
              message: "empty response",
            });
          }
        }
      });
    }, bot.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const deleteMessage = (msgId?: string) => {
    botStore.updateBotSession(
      (session) =>
        (session.messages = session.messages.filter((m) => m.id !== msgId)),
      bot.id,
    );
  };

  const onDelete = (msgId: string) => {
    deleteMessage(msgId);
  };

  const context: ChatMessage[] = useMemo(() => {
    return bot.hideContext ? [] : bot.context.slice();
  }, [bot.context, bot.hideContext]);

  const getUrlTypePrefix = (type: string) => {
    if (type === "text/html") return "HTML";
    if (type === "application/pdf") return "PDF";
    if (type === "text/plain") return "TXT";
    return Locale.Upload.UnknownFileType;
  };

  // preview messages
  const renderMessages = useMemo(() => {
    const getFrontendMessages = (messages: ChatMessage[]) => {
      return messages.map((message) => {
        if (!message.urlDetail || message.urlDetail.type === "image/jpeg")
          return message;
        const urlTypePrefix = getUrlTypePrefix(message.urlDetail.type);
        const sizeInKB = Math.round(message.urlDetail.size / 1024);
        return {
          ...message,
          content: `${message.urlDetail.url}\n\`${urlTypePrefix} • ${sizeInKB} KB\``,
        };
      });
    };

    const getUrlPreviewMessage = () => {
      const lastMessage = session.messages[session.messages.length - 1];
      const showPreviewUrl = temporaryURLInput && !lastMessage?.streaming;
      let previewUrlMessage: ChatMessage | undefined;

      if (showPreviewUrl) {
        previewUrlMessage = createMessage({
          role: "user",
          content: `${temporaryURLInput}\n\`${Locale.Chat.LoadingURL}\``,
        });
      }
      return previewUrlMessage;
    };

    return context
      .concat(
        bot.botHello
          ? [
              createMessage({
                role: "assistant",
                content: bot.botHello,
              }),
            ]
          : [],
      )
      .concat(getFrontendMessages(session.messages))
      .concat(getUrlPreviewMessage() || []);
  }, [session.messages, bot.botHello, temporaryURLInput, context]);

  const [msgRenderIndex, _setMsgRenderIndex] = useState(
    Math.max(0, renderMessages.length - CHAT_PAGE_SIZE),
  );
  function setMsgRenderIndex(newIndex: number) {
    newIndex = Math.min(renderMessages.length - CHAT_PAGE_SIZE, newIndex);
    newIndex = Math.max(0, newIndex);
    _setMsgRenderIndex(newIndex);
  }

  const messages = useMemo(() => {
    const endRenderIndex = Math.min(
      msgRenderIndex + 3 * CHAT_PAGE_SIZE,
      renderMessages.length,
    );
    return renderMessages.slice(msgRenderIndex, endRenderIndex);
  }, [msgRenderIndex, renderMessages]);

  const onChatBodyScroll = (e: HTMLElement) => {
    const bottomHeight = e.scrollTop + e.clientHeight;
    const edgeThreshold = e.clientHeight;

    const isTouchTopEdge = e.scrollTop <= edgeThreshold;
    const isTouchBottomEdge = bottomHeight >= e.scrollHeight - edgeThreshold;
    const isHitBottom = bottomHeight >= e.scrollHeight - 10;

    const prevPageMsgIndex = msgRenderIndex - CHAT_PAGE_SIZE;
    const nextPageMsgIndex = msgRenderIndex + CHAT_PAGE_SIZE;

    if (isTouchTopEdge && !isTouchBottomEdge) {
      setMsgRenderIndex(prevPageMsgIndex);
    } else if (isTouchBottomEdge) {
      setMsgRenderIndex(nextPageMsgIndex);
    }

    setAutoScroll(isHitBottom);
  };

  function scrollToBottom() {
    setMsgRenderIndex(renderMessages.length - CHAT_PAGE_SIZE);
    scrollDomToBottom();
  }

  // clear context index = context length + index in messages
  const clearContextIndex =
    (session.clearContextIndex ?? -1) >= 0
      ? session.clearContextIndex! +
        context.length +
        (bot.botHello ? 1 : 0) -
        msgRenderIndex
      : -1;

  const clearContext = () => {
    botStore.updateBotSession((session) => {
      if (session.clearContextIndex === session.messages.length) {
        session.clearContextIndex = undefined;
      } else {
        session.clearContextIndex = session.messages.length;
      }
    }, bot.id);
  };
  const stop = () => ChatControllerPool.stop(bot.id);
  const isRunning = ChatControllerPool.isRunning(bot.id);

  // processing websocket streamed messages
  // Function to process the content and extract messages
  const processContent = (messageString: string) => {
    // Initialize variables for the current message and the list of all messages
    let currentMessage = { innerThoughts: "", assistantMessage: "", functionCall: "" };
    const messageList = [];

    // Helper function to reset the current message
    const resetCurrentMessage = () => ({ innerThoughts: "", assistantMessage: "", functionCall: "" });

    // Helper function to add the current message to the list and reset it
    const finalizeMessage = () => {
        messageList.push({ ...currentMessage });
        currentMessage = resetCurrentMessage();
    };

    // Split the main text into lines
    const lines = messageString.split("\n");

    // Process each line and extract the relevant parts
    lines.forEach(line => {
        if (line.startsWith("internal_monologue:")) {
            // If 'innerThoughts' is already set, finalize current message
            if (currentMessage.innerThoughts !== "") {
                finalizeMessage();
            }
            currentMessage.innerThoughts += line.replace("internal_monologue:", "").trim() + " ";
        } else if (line.startsWith("assistant_message:")) {
            // If 'assistantMessage' is already set, finalize current message
            if (currentMessage.assistantMessage !== "") {
                finalizeMessage();
            }
            currentMessage.assistantMessage += line.replace("assistant_message:", "").trim() + " ";
        } else if (line.startsWith("function_call:")) {
            // If 'functionCall' is already set, finalize current message
            if (currentMessage.functionCall !== "") {
                finalizeMessage();
            }
            currentMessage.functionCall += line.replace("function_call:", "").trim() + " ";
        }
    });

    // Add the last message to the list if it's not empty
    if (currentMessage.innerThoughts !== "" || currentMessage.assistantMessage !== "" || currentMessage.functionCall !== "") {
        messageList.push(currentMessage);
    }

    return messageList;
  }

  const messageToMainContentString = (messageObj: any) => {
    const assistantMessage = messageObj.assistantMessage.trim();
    const functionCall = messageObj.functionCall.trim();

    // Prepare the main message based on the availability of the function call:
    const mainMessage = (assistantMessage && functionCall && !functionCall.startsWith('send_message(')) ?
      `${assistantMessage}\n` + // Newline outside of code block
      "```python\n" + `${functionCall}\n` + "```"       // Newline inside code block
    : (functionCall && !functionCall.startsWith('send_message(')) ?
      "```python\n" + `${functionCall}\n` + "```"       // Newline inside code block
    : (assistantMessage) ?
      assistantMessage
    // : mainText;
    : "";
    return mainMessage;
  }

  return (
    <div className="flex flex-col relative h-full" key={bot.id}>
      <ChatHeader />
      <ScrollArea
        className="flex-1 overflow-auto overflow-x-hidden relative overscroll-none pb-10 p-5"
        ref={scrollRef}
        onScroll={(e) => onChatBodyScroll(e.currentTarget)}
        onMouseDown={() => inputRef.current?.blur()}
        onTouchStart={() => {
          inputRef.current?.blur();
          setAutoScroll(false);
        }}
      >
        <div className="space-y-5">
          {messages.map((message, i) => {
            const isUser = message.role === "user";
            const isMemory = message.role === "memory";
            const isContext = i < context.length;
            const showActions =
              i > 0 && !(message.content.length === 0) && !isContext;
            const showThinking = message.streaming;
            const shouldShowClearContextDivider = i === clearContextIndex - 1;

            // Logic to split the message content into main text and inner thoughts
            // const innerThoughtsPattern = /.../; // Define your pattern here
            // const splitContent = message.content.split(innerThoughtsPattern);
            // const mainText = splitContent[0];
            // const innerThoughts = splitContent[1]; // Assuming there's only one inner thought per message

            // Logic to extract the inner monologue and assistant message
            // const innerMonologuePrefix = "internal_monologue:";
            // const assistantMessagePrefix = "assistant_message:";
            // let mainText = message.content;
            // let innerThoughts = null;

            // // Check for inner monologue and split the text accordingly
            // if (mainText.includes(innerMonologuePrefix)) {
            //   const parts = mainText.split("\n").map(part => part.trim());
            //   const innerMonologuePart = parts.find(part => part.startsWith(innerMonologuePrefix));
            //   const assistantMessagePart = parts.find(part => part.startsWith(assistantMessagePrefix));

            //   if (innerMonologuePart) {
            //     innerThoughts = innerMonologuePart.replace(innerMonologuePrefix, '').trim();
            //   }
            //   if (assistantMessagePart) {
            //     mainText = assistantMessagePart.replace(assistantMessagePrefix, '').trim();
            //   }
            // }
            let mainText = message.content;
            const isAssistantMessage = /internal_monologue:|assistant_message:|function_call:/.test(message.content);
            // const isAssistantMessage = !isUser;
            const messageList = isAssistantMessage ? 
              processContent(mainText).map((v) => ({innerThoughts: v.innerThoughts, mainMessage: messageToMainContentString(v)})) 
              : [{innerThoughts: null, mainMessage: mainText}];
            // const innerThoughts = isAssistantMessage ? messageList[0].innerThoughts : null;
            // const mainMessage = isAssistantMessage ? messageToMainContentString(messageList[0]) : messageList[0];

            // // Initialize placeholders for the different parts of the message
            // let innerThoughts = "";
            // let assistantMessage = "";
            // let functionCall = "";

            // // Split the main text into lines
            // const lines = message.content.split("\n");
            // const messageList = [];

            // // Process each line and extract the relevant parts
            // lines.forEach(line => {
            //   // If we ever hit a duplicate field, start the next message
            //   if (line.startsWith("internal_monologue:")) {
            //     innerThoughts += line.replace("internal_monologue:", "").trim() + " ";
            //   } else if (line.startsWith("assistant_message:")) {
            //     assistantMessage += line.replace("assistant_message:", "").trim() + " ";
            //   } else if (line.startsWith("function_call:")) {
            //     functionCall += line.replace("function_call:", "").trim() + " ";
            //     // functionCall = JSON.parse(functionCall);
            //   }
            // });

            // Trim the strings to remove any excess whitespace from the ends
            // innerThoughts = innerThoughts.trim();
            // assistantMessage = assistantMessage.trim();
            // functionCall = functionCall.trim();

            // Prepare the main message based on the availability of the function call:
            // const mainMessage = (assistantMessage && functionCall && !functionCall.startsWith('send_message(')) ?
            //   `${assistantMessage}\n` + // Newline outside of code block
            //   "```python\n" + `${functionCall}\n` + "```"       // Newline inside code block
            // : (functionCall && !functionCall.startsWith('send_message(')) ?
            //   "```python\n" + `${functionCall}\n` + "```"       // Newline inside code block
            // : (assistantMessage) ?
            //   assistantMessage
            // : mainText;

            // Inline styling for inner thoughts
            const innerThoughtsStyle = {
              fontStyle: 'italic',
              // color: '#666',
              // position: 'absolute',
              // top: '-20px',
              // right: '0',
              // Other styles as needed
            };

            // console.log(`mainText: ${mainText}`);
            // console.log(`innerThoughts: ${innerThoughts}`);

            return (
              <div className="space-y-5" key={i}>
                <div
                  className={
                    isUser
                      ? "flex flex-row-reverse"
                      : "flex flex-row last:animate-[slide-in_ease_0.3s]"
                  }
                >
                  <HoverCard openDelay={200}>
                    <HoverCardTrigger asChild>
                      <div
                        className={cn(
                          "max-w-[80%] flex flex-col items-start",
                          isUser && "items-end",
                        )}
                      >
                        {showThinking && (
                          <div
                            className={
                              "text-xs text-[#aaa] leading-normal my-1"
                            }
                          >
                            {Locale.Chat.Thinking}
                          </div>
                        )}

                      {messageList.map((messageObj, index) => (
                        <div>

                        {messageObj.innerThoughts && (
                          <div
                            className={
                              "text-xs text-[#aaa] leading-normal my-1"
                            }
                            style={innerThoughtsStyle}
                          >
                           💭 {messageObj.innerThoughts}
                          </div>
                        )}
                        <div
                          className={cn(
                            "box-border max-w-full text-sm select-text relative break-words rounded-lg px-3 py-2",
                            isUser
                              ? "ml-auto bg-primary text-primary-foreground"
                              : isMemory
                              ? "italic text-secondary-foreground"
                              : "bg-muted",
                          )}
                        >
                          {message.urlDetail?.type === "image/jpeg" && (
                            <img
                              src={message.urlDetail.url}
                              alt="Message image"
                              className="object-contain w-full h-52 rounded-lg mb-2"
                            />
                          )}
                          <Markdown
                            content={messageObj.mainMessage}
                            loading={
                              message.streaming &&
                              message.content.length === 0 &&
                              !isUser
                            }
                            onDoubleClickCapture={() => {
                              if (!isMobileScreen) return;
                              setUserInput(message.content);
                            }}
                            parentRef={scrollRef}
                            defaultShow={i >= messages.length - 6}
                          />
                        </div>

                        </div>))}

                        <div className="text-xs text-muted-foreground opacity-80 whitespace-nowrap text-right w-full box-border pointer-events-none z-[1]">
                          {isContext
                            ? Locale.Chat.IsContext
                            : message.date?.toLocaleString()}
                        </div>
                      </div>
                    </HoverCardTrigger>

                    {showActions && (
                      <HoverCardContent
                        side="top"
                        align={isUser ? "end" : "start"}
                        className="py-1 px-0 w-fit"
                      >
                        <div className="flex items-center divide-x">
                          {!message.streaming && (
                            <>
                              {message.id && (
                                <ChatAction
                                  text={Locale.Chat.Actions.Delete}
                                  icon={<Trash className="w-4 h-4" />}
                                  onClick={() => onDelete(message.id!)}
                                />
                              )}
                              <ChatAction
                                text={Locale.Chat.Actions.Copy}
                                icon={<Clipboard className="w-4 h-4" />}
                                onClick={() =>
                                  copyToClipboard(message.content, toast)
                                }
                              />
                            </>
                          )}
                        </div>
                      </HoverCardContent>
                    )}
                  </HoverCard>
                </div>
                {shouldShowClearContextDivider && (
                  <ClearContextDivider botId={bot.id} />
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>
      <Separator />
      <div className="relative w-full box-border flex-col pt-2.5 p-5 space-y-2">
        <div className="flex justify-between items-center">
          <ChatAction
            text={Locale.Chat.InputActions.Clear}
            icon={<Eraser className="w-4 h-4" />}
            onClick={clearContext}
            showTitle
            buttonVariant="outline"
          />
          {isRunning && (
            <ChatAction
              onClick={stop}
              text={Locale.Chat.InputActions.Stop}
              icon={<PauseCircle className="w-4 h-4" />}
              showTitle
              buttonVariant="outline"
            />
          )}
        </div>

        <ChatInput
          inputRef={inputRef}
          userInput={userInput}
          temporaryURLInput={temporaryURLInput}
          setUserInput={setUserInput}
          setTemporaryURLInput={setTemporaryURLInput}
          scrollToBottom={scrollToBottom}
          setAutoScroll={setAutoScroll}
        />
      </div>
    </div>
  );
}
