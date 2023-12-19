import { REQUEST_TIMEOUT_MS } from "@/app/constant";

import { fetchEventSource } from "@fortaine/fetch-event-source";
import { Embedding } from "../fetch/url";

export const MESSAGE_ROLES = [
  "system",
  "user",
  "assistant",
  "URL",
  "memory",
] as const;
export type MessageRole = (typeof MESSAGE_ROLES)[number];

export interface MessageContentDetail {
  type: "text" | "image_url";
  text: string;
  image_url: { url: string };
}

export type MessageContent = string | MessageContentDetail[];

export interface RequestMessage {
  role: MessageRole;
  content: MessageContent;
}

export interface ResponseMessage {
  role: MessageRole;
  content: string;
}

export const ALL_MODELS = [
  "gpt-4",
  "gpt-4-1106-preview",
  "gpt-4-vision-preview",
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-16k",
] as const;

export type ModelType = (typeof ALL_MODELS)[number];

export interface LLMConfig {
  model: ModelType;
  temperature?: number;
  topP?: number;
  sendMemory?: boolean;
  maxTokens?: number;
}

export interface ChatOptions {
  message: MessageContent;
  chatHistory: RequestMessage[];
  config: LLMConfig;
  datasource?: string;
  embeddings?: Embedding[];
  agent_name?: string;
  controller: AbortController;
  onUpdate: (message: string) => void;
  onFinish: (memoryMessage?: ResponseMessage) => void;
  onError?: (err: Error) => void;
}

const CHAT_PATH = "/api/llm";

export function isVisionModel(model: ModelType) {
  return model === "gpt-4-vision-preview";
}

export class LLMApi {
  async chat(options: ChatOptions) {
    const requestPayload = {
      message: options.message,
      chatHistory: options.chatHistory.map((m) => ({
        role: m.role,
        content: m.content,
      })),
      config: options.config,
      datasource: options.datasource,
      embeddings: options.embeddings,
      agent_name: options.agent_name,
    };

    console.log("[Request] payload: ", requestPayload);

    const requestTimeoutId = setTimeout(
      () => options.controller?.abort(),
      REQUEST_TIMEOUT_MS,
    );

    options.controller.signal.onabort = () => options.onFinish();
    const handleError = (e: any) => {
      clearTimeout(requestTimeoutId);
      console.log("[Request] failed to make a chat request", e);
      options.onError?.(e as Error);
    };

    try {
      const chatPayload = {
        method: "POST",
        body: JSON.stringify(requestPayload),
        signal: options.controller?.signal,
        headers: {
          "Content-Type": "application/json",
        },
      };

      let llmResponse = "";
      await fetchEventSource(CHAT_PATH, {
        ...chatPayload,
        async onopen(res) {
          clearTimeout(requestTimeoutId);
          if (!res.ok) {
            const json = await res.json();
            handleError(new Error(json.message));
          }
        },
        onmessage(msg) {
          try {
            const json = JSON.parse(msg.data);
            if (json.done) {
              options.onFinish(json.memoryMessage);
            } else if (json.error) {
              options.onError?.(new Error(json.error));
            } else {
              // received a new token
              llmResponse += json;
              options.onUpdate(llmResponse);
            }
          } catch (e) {
            console.error("[Request] error parsing streaming delta", msg);
          }
        },
        onclose() {
          options.onFinish();
        },
        onerror: handleError,
        openWhenHidden: true,
      });
    } catch (e) {
      handleError(e);
    }
  }
}
