const en = {
  Chat: {
    SubTitle: (count: number) => `${count} messages`,
    Actions: {
      ChatList: "Go To Chat List",
      Copy: "Copy",
      Delete: "Delete",
    },
    InputActions: {
      Stop: "Stop generating",
      Clear: "Clear Context",
    },
    Thinking: "Thinking…",
    InputMobile: "Enter to send",
    Input:
      "Enter to send, Shift + Enter to wrap, enter URLs to add a PDF or HTML document to the context",
    Send: "Send",
    IsContext: "Contextual Prompt",
    LoadingURL: "Loading content...",
    LLMError:
      "There was an error calling the OpenAI API. Please try again later.",
    MemGPTError: "Oh no... there was an error calling the MemGPT server.",
  },
  Upload: {
    Invalid: (acceptTypes: string) =>
      `Invalid file type. Please select a file with one of these formats: ${acceptTypes}`,
    SizeExceeded: (limitSize: number) =>
      `File size exceeded. Limit is ${limitSize} MB`,
    Failed: (e: string) => `Error uploading file: ${e}`,
    ParseDataURLFailed: "Unable to read file: Please check if it's encrypted.",
    UnknownFileType: "Unknown file type",
    ModelDoesNotSupportImages: (acceptTypes: string) =>
      `Image upload is not supported for this model. Upload one of the supported types instead: ${acceptTypes}`,
  },
  Export: {
    Image: {
      Modal: "Long press or right click to save image",
    },
  },
  Memory: {
    Title: "Memory Prompt",
    Send: "Send Memory",
  },
  Home: {
    Github: "Github",
    Logout: "Logout",
    Settings: "Settings",
  },
  Settings: {
    Title: "Settings",
    SubTitle: "All Settings",
    Danger: {
      Clear: {
        Title: "Clear All Data",
        // SubTitle: "Reset all bots and clear all messages",
        SubTitle: "Reset all agents and clear all messages",
        Action: "Clear",
        Confirm: "Confirm to clear all data?",
      },
    },

    Model: "Model",
    Temperature: {
      Title: "Temperature",
      SubTitle: "A larger value makes the more random output",
    },
    TopP: {
      Title: "Top P",
      SubTitle: "Do not alter this value together with temperature",
    },
    MaxTokens: {
      Title: "Max Tokens",
      SubTitle: "Maximum length of input tokens and generated tokens",
    },
    Backup: {
      Download: {
        // Title: "Backup Bots",
        Title: "Backup Agents",
        SutTitle: "Download the state of your agents to a JSON file",
      },
      Upload: {
        // Title: "Restore Bots",
        Title: "Restore Agents",
        // SutTitle: "Upload the state of your bots from a JSON file",
        SutTitle: "Upload the state of your agents from a JSON file",
        // Success: "Successfully restored the bots from the JSON file",
        Success: "Successfully restored the agents from the JSON file",
        Failed: (e: string) => `Error importing the JSON file: ${e}`,
      },
    },
  },
  Store: {
    // DefaultBotName: "New Bot",
    DefaultBotName: "New Agent",
    BotHello: "Hello! How can I assist you today?",
  },
  Copy: {
    Success: "Copied to clipboard",
    Failed: "Copy failed, please grant permission to access clipboard",
  },
  Context: {
    Add: "Add a Prompt",
    Clear: "Context Cleared",
    Revert: "Revert",
    Title: "Context Prompt Settings",
  },
  Share: {
    // Title: "Share Bot",
    Title: "Share Agent",
    Url: {
      Title: "URL",
      // Hint: "Use the URL to share your bot. The URL will be valid for 30 days.",
      Hint: "Use the URL to share your agent. The URL will be valid for 30 days.",
      Error: "Oops, something went wrong. Please try again later.",
    },
  },
  Bot: {
    // Name: "Bot",
    Name: "Agent",
    Page: {
      Search: (count: number) => `Search agents - ${count} total`,
      // Create: "Create bot",
      Create: "Create new agent",
    },
    Item: {
      Edit: "Edit",
      Delete: "Delete",
      DeleteConfirm: "Confirm to delete?",
      Share: "Share",
    },
    EditModal: {
      // Title: `Edit Bot`,
      Title: `Edit Agent`,
      Clone: "Clone",
    },
    Config: {
      // Avatar: "Bot Avatar",
      Avatar: "Agent Avatar",
      // Name: "Bot Name",
      Name: "Agent Name",
      HideContext: {
        Title: "Hide Context Prompts",
        SubTitle: "Do not show in-context prompts in chat",
      },
      BotHello: {
        Title: "Welcome Message",
        SubTitle: "Welcome message sent when starting a new chat",
      },
      // Title: "Bot Settings",
      Title: "Agent Settings",
    },
  },

  Welcome: {
    // Title: "LlamaIndex Chat",
    Title: "MemGPT",
    SubTitle: "Chat with MemGPT",
    Quote:
      "“This tool has saved me countless hours of work and helped me apply AI features to my work faster than ever before.”",
    LoginLinkedinTitle: "Login with LinkedIn",
  },
};

export type LocaleType = typeof en;

export default en;
