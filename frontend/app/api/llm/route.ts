import {
  ChatHistory,
  ChatMessage,
  DefaultContextGenerator,
  HistoryChatEngine,
  IndexDict,
  OpenAI,
  ServiceContext,
  SimpleChatHistory,
  SummaryChatHistory,
  TextNode,
  VectorStoreIndex,
  serviceContextFromDefaults,
} from "llamaindex";
import { NextRequest, NextResponse } from "next/server";
import { LLMConfig, MessageContent } from "@/app/client/platforms/llm";
import { getDataSource } from "./datasource";
import {
  DATASOURCES_CHUNK_OVERLAP,
  DATASOURCES_CHUNK_SIZE,
} from "@/scripts/constants.mjs";
import { Embedding } from "@/app/client/fetch/url";
import Locale from "@/app/locales";

async function createChatEngine(
  serviceContext: ServiceContext,
  datasource?: string,
  embeddings?: Embedding[],
) {
  let contextGenerator;
  if (datasource || embeddings) {
    let index;
    if (embeddings) {
      // TODO: merge indexes, currently we prefer own embeddings
      index = await createIndex(serviceContext, embeddings);
    } else if (datasource) {
      index = await getDataSource(serviceContext, datasource);
    }
    const retriever = index!.asRetriever();
    retriever.similarityTopK = 5;

    contextGenerator = new DefaultContextGenerator({ retriever });
  }

  return new HistoryChatEngine({
    llm: serviceContext.llm,
    contextGenerator,
  });
}

async function createIndex(
  serviceContext: ServiceContext,
  embeddings: Embedding[],
) {
  const embeddingResults = embeddings.map((config) => {
    return new TextNode({ text: config.text, embedding: config.embedding });
  });
  const indexDict = new IndexDict();
  for (const node of embeddingResults) {
    indexDict.addNode(node);
  }

  const index = await VectorStoreIndex.init({
    indexStruct: indexDict,
    serviceContext: serviceContext,
  });

  index.vectorStore.add(embeddingResults);
  if (!index.vectorStore.storesText) {
    await index.docStore.addDocuments(embeddingResults, true);
  }
  await index.indexStore?.addIndexStruct(indexDict);
  index.indexStruct = indexDict;
  return index;
}

function condition_to_stop_receiving(data: any) {
  if (
    data.type === "command_response" ||
    data.type === "agent_response_end" ||
    data.type === "agent_response_error" ||
    data.type === "server_error"
  ) {
    return true;
  } else {
    return false;
  }
}

function format_agent_response(response: any) {
  var message;
  if (response.message_type === "internal_monologue") {
    message = `💭 _${response.message}_`;
  } else if (response.message_type === "assistant_message") {
    message = `${response.message}`;
  } else if (response.message_type === "function_message") {
    message = null;
  }
  return message;
}

function createReadableStreamFromWebSocket(
  agent_name: string | undefined,
  message: MessageContent,
): ReadableStream<string> {
  // Initialize the TransformStream
  let responseStream = new TransformStream();
  const writer = responseStream.writable.getWriter();
  const encoder = new TextEncoder();

  // Create a WebSocket connection
  const websocketUrl = "ws://localhost:8282";
  const socket = new WebSocket(websocketUrl);

  socket.onopen = () => {
    // Send the message as a JSON string when the WebSocket connection is opened
    // socket.send(JSON.stringify({ 'message': message }));
    // socket.send(JSON.stringify({
    //   type: 'command',
    //   command: 'load_agent',
    //   name: 'agent_26'
    // }));
    console.log(`[socket.onopen]`);
    socket.send(
      JSON.stringify({
        type: "user_message",
        message: `${String(message)}`,
        // agent_name: 'agent_26',
        agent_name: agent_name,
      }),
    );
  };

  socket.onmessage = (event) => {
    // Handle incoming messages
    console.log(`[socket.onmessage]`);
    try {
      const data = JSON.parse(event.data);
      console.log("Got back data");
      console.log(JSON.stringify(data));

      if (condition_to_stop_receiving(data)) {
        // Write an error
        if (data.type === "agent_response_error" || data.type === "server_error") {
            writer.write(
              `data: ${JSON.stringify({
                // error: Locale.Chat.LLMError,
                error: JSON.stringify(data),
              })}\n\n`,
            );
        }
        // The websocket said it will stop sending data so close
        writer.write(
          encoder.encode(
            `data: ${JSON.stringify({
              done: true,
            })}\n\n`,
          ),
        );
        writer.close();
      } else {
        // Write the data to the stream
        if (data.type === "agent_response") {
          const message = format_agent_response(data);
          if (message !== null) {
            writer.write(encoder.encode(`data: "${message}\\n"\n\n`));
          }
        }
      }
    } catch (error) {
      console.error("[WebSocket Error]", error);
      writer.write(
        encoder.encode(
          `data: ${JSON.stringify({
            error: "Error processing WebSocket message",
          })}\n\n`,
        ),
      );
      writer.close();
    }
  };

  socket.onerror = (error) => {
    // Handle WebSocket errors
    console.log(`[socket.onerror]`);
    console.error("[WebSocket Error]", error);
    writer.write(
      encoder.encode(
        `data: ${JSON.stringify({ error: "WebSocket connection error" })}\n\n`,
      ),
    );
    writer.close();
  };

  socket.onclose = () => {
    // Close the stream when the WebSocket is closed
    console.log(`[socket.onclose]`);
    writer.close();
  };

  return responseStream.readable;
}

function createReadableStream(
  stream: AsyncGenerator<string, void, unknown>,
  chatHistory: ChatHistory,
) {
  let responseStream = new TransformStream();
  const writer = responseStream.writable.getWriter();
  let aborted = false;
  writer.closed.catch(() => {
    // reader aborted the stream
    aborted = true;
  });
  const encoder = new TextEncoder();
  const onNext = async () => {
    try {
      const { value, done } = await stream.next();
      if (aborted) return;
      if (!done) {
        console.log(`data: ${JSON.stringify(value)}\n\n`);
        writer.write(encoder.encode(`data: ${JSON.stringify(value)}\n\n`));
        onNext();
      } else {
        console.log(
          `data: ${JSON.stringify({
            done: true,
            // get the optional message containing the chat summary
            memoryMessage: chatHistory
              .newMessages()
              .filter((m) => m.role === "memory")
              .at(0),
          })}\n\n`,
        );
        writer.write(
          `data: ${JSON.stringify({
            done: true,
            // get the optional message containing the chat summary
            memoryMessage: chatHistory
              .newMessages()
              .filter((m) => m.role === "memory")
              .at(0),
          })}\n\n`,
        );
        writer.close();
      }
    } catch (error) {
      console.error("[LlamaIndex]", error);
      writer.write(
        `data: ${JSON.stringify({
          // error: Locale.Chat.LLMError,
          error: Locale.Chat.MemGPTError,
        })}\n\n`,
      );
      writer.close();
    }
  };
  onNext();
  return responseStream.readable;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      message,
      chatHistory: messages,
      datasource,
      agent_name,
      config,
      embeddings,
    }: {
      message: MessageContent;
      chatHistory: ChatMessage[];
      datasource: string | undefined;
      agent_name: string | undefined;
      config: LLMConfig;
      embeddings: Embedding[] | undefined;
    } = body;
    if (!message || !messages || !config) {
      return NextResponse.json(
        {
          error:
            "message, chatHistory and config are required in the request body",
        },
        { status: 400 },
      );
    }
    console.log(`config: ${JSON.stringify(config)}`);
    console.log(`messages: ${JSON.stringify(messages)}`);
    console.log(`datasource: ${JSON.stringify(datasource)}`);
    console.log(`agent_name: ${JSON.stringify(agent_name)}`);
    // console.log(`messages: ${messages}`)

    // const USE_OPENAI = true;
    const USE_OPENAI = false;

    var readableStream;
    if (USE_OPENAI) {
      // Original way OpenAI POST SSE is getting called
      const llm = new OpenAI({
        model: config.model,
        temperature: config.temperature,
        topP: config.topP,
        maxTokens: config.maxTokens,
      });

      const serviceContext = serviceContextFromDefaults({
        llm,
        chunkSize: DATASOURCES_CHUNK_SIZE,
        chunkOverlap: DATASOURCES_CHUNK_OVERLAP,
      });

      const chatEngine = await createChatEngine(
        serviceContext,
        datasource,
        embeddings,
      );
      const chatHistory = config.sendMemory
        ? new SummaryChatHistory({ llm, messages })
        : new SimpleChatHistory({ messages });

      const stream = await chatEngine.chat(message, chatHistory, true);
      readableStream = createReadableStream(stream, chatHistory);
    } else {
      // Uses a websocket inside the readable stream
      readableStream = createReadableStreamFromWebSocket(agent_name, message);
    }

    return new NextResponse(readableStream, {
      headers: {
        "Content-Type": "text/event-stream",
        Connection: "keep-alive",
        "Cache-Control": "no-cache, no-transform",
      },
    });
  } catch (error) {
    console.error("[LlamaIndex]", error);
    return NextResponse.json(
      {
        error: Locale.Chat.MemGPTError,
      },
      {
        status: 500,
      },
    );
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Set max running time of function, for Vercel Hobby use 10 seconds, see https://vercel.com/docs/functions/serverless-functions/runtimes#maxduration
export const maxDuration = 120;
