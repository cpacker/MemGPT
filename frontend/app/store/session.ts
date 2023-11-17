import { nanoid } from "nanoid";
import { ChatControllerPool } from "../client/controller";
import {
  Embedding,
  URLDetail,
  URLDetailContent,
  fetchSiteContent,
  isURL,
} from "../client/fetch/url";
import {
  MessageContentDetail,
  LLMApi,
  RequestMessage,
  MessageRole,
  ResponseMessage,
} from "../client/platforms/llm";
import { prettyObject } from "../utils/format";
import { Bot } from "./bot";

export type ChatMessage = {
  role: MessageRole;
  content: string;
  date?: string;
  streaming?: boolean;
  isError?: boolean;
  id?: string;
  urlDetail?: URLDetail;
};

export function createMessage(override: Partial<ChatMessage>): ChatMessage {
  return {
    id: nanoid(),
    date: new Date().toLocaleString(),
    role: "user",
    content: "",
    ...override,
  };
}

export interface ChatSession {
  messages: ChatMessage[];
  clearContextIndex?: number;
}

export function createEmptySession(): ChatSession {
  return {
    messages: [],
  };
}

async function createTextInputMessage(
  content: string,
  urlDetail?: URLDetailContent,
): Promise<ChatMessage> {
  if (isURL(content)) {
    const urlDetail = await fetchSiteContent(content);
    return createFileInputMessage(urlDetail);
  } else {
    return createMessage({
      role: "user",
      content,
      urlDetail,
    });
  }
}

async function createFileInputMessage(
  fileDetail: URLDetailContent,
): Promise<ChatMessage> {
  console.log("[User Input] did get file detail: ", fileDetail);
  delete fileDetail["content"]; // clean content in file detail as we are only going to use its embeddings
  return createMessage({
    role: "user",
    urlDetail: fileDetail,
  });
}

function transformAssistantMessageForSending(
  message: ChatMessage,
): RequestMessage {
  const { content } = message;
  // messages with role URL are assistant messages that contain a URL - the content is already retrieved by context-prompt.tsx
  if (message.role !== "URL") return message;
  return {
    role: "assistant",
    content,
  };
}

async function createUserMessage(
  content?: string,
  urlDetail?: URLDetailContent,
): Promise<ChatMessage> {
  let userMessage: ChatMessage;
  if (content) {
    userMessage = await createTextInputMessage(content, urlDetail);
  } else if (urlDetail) {
    userMessage = await createFileInputMessage(urlDetail);
  } else {
    throw new Error("Invalid user message");
  }
  return userMessage;
}

export async function callSession(
  bot: Bot,
  session: ChatSession,
  callbacks: {
    onUpdateMessages: (messages: ChatMessage[]) => void;
  },
  content?: string,
  fileDetail?: URLDetailContent,
): Promise<void> {
  const modelConfig = bot.modelConfig;

  let userMessage: ChatMessage;

  try {
    userMessage = await createUserMessage(content, fileDetail);
  } catch (error: any) {
    // an error occurred when creating user message, show error message as bot message and don't call API
    const userMessage = createMessage({
      role: "user",
      content,
    });
    const botMessage = createMessage({
      role: "assistant",
      content: prettyObject({
        error: true,
        message: error.message || "Invalid user message",
      }),
    });
    // updating the session will trigger a re-render, so it will display the messages
    session.messages = session.messages.concat([userMessage, botMessage]);
    callbacks.onUpdateMessages(session.messages);
    return;
  }

  const botMessage: ChatMessage = createMessage({
    role: "assistant",
    streaming: true,
  });

  const contextPrompts = bot.context.slice();
  // get messages starting from the last clear context index (or all messages if there is no clear context index)
  const recentMessages = !session.clearContextIndex
    ? session.messages
    : session.messages.slice(session.clearContextIndex);
  let sendMessages = [
    ...contextPrompts,
    ...recentMessages.map(transformAssistantMessageForSending),
  ];

  // save user's and bot's message
  session.messages = session.messages.concat([userMessage, botMessage]);
  callbacks.onUpdateMessages(session.messages);

  let embeddings: Embedding[] | undefined;
  let message;
  if (userMessage.urlDetail && userMessage.urlDetail.type !== "image/jpeg") {
    // if the user sends document, let the LLM summarize the content of the URL and just use the document's embeddings
    message = "Summarize the given context briefly in 200 words or less";
    embeddings = userMessage.urlDetail?.embeddings;
    sendMessages = [];
  } else {
    // collect embeddings of all messages
    embeddings = session.messages
      .flatMap((message: ChatMessage) => message.urlDetail?.embeddings)
      .filter((m) => m !== undefined) as Embedding[];
    embeddings = embeddings.length > 0 ? embeddings : undefined;
    if (userMessage.urlDetail?.type === "image/jpeg") {
      message = [
        {
          type: "text",
          text: userMessage.content,
        } as MessageContentDetail,
        {
          type: "image_url",
          image_url: {
            url: userMessage.urlDetail.url,
          },
        } as MessageContentDetail,
      ];
    } else {
      message = userMessage.content;
    }
  }

  // make request
  const controller = new AbortController();
  ChatControllerPool.addController(bot.id, controller);
  const api = new LLMApi();
  await api.chat({
    agent_name: bot.name,
    datasource: bot.datasource,
    embeddings,
    message: message,
    chatHistory: sendMessages,
    config: modelConfig,
    controller,
    onUpdate(message) {
      if (message) {
        botMessage.content = message;
        callbacks.onUpdateMessages(session.messages.concat());
      }
    },
    onFinish(memoryMessage?: ResponseMessage) {
      botMessage.streaming = false;
      if (memoryMessage) {
        // all optional memory message returned by the LLM
        const newChatMessages = createMessage({ ...memoryMessage });
        session.messages = session.messages.concat(newChatMessages);
      }
      callbacks.onUpdateMessages(session.messages.concat());
      ChatControllerPool.remove(bot.id);
    },
    onError(error) {
      const isAborted = error.message.includes("aborted");
      botMessage.content +=
        "\n\n" +
        prettyObject({
          error: true,
          message: error.message,
        });
      botMessage.streaming = false;
      userMessage.isError = !isAborted;
      botMessage.isError = !isAborted;
      callbacks.onUpdateMessages(session.messages);
      ChatControllerPool.remove(bot.id);

      console.error("[Chat] failed ", error);
    },
  });
}
