export const GITHUB_URL = "https://github.com/cpacker/MemGPT";

export enum Path {
  Home = "/",
  Chat = "/",
  Settings = "/settings",
  Bots = "/",
}

export enum FileName {
  Bots = "bots.json",
}

export const REQUEST_TIMEOUT_MS = 60000;

export const CHAT_PAGE_SIZE = 15;
export const MAX_RENDER_MSG_COUNT = 45;

export const ALLOWED_IMAGE_EXTENSIONS = ["jpeg", "jpg"];
export const ALLOWED_TEXT_EXTENSIONS = ["pdf", "txt"];
export const ALLOWED_DOCUMENT_EXTENSIONS = [
  ...ALLOWED_TEXT_EXTENSIONS,
  ...ALLOWED_IMAGE_EXTENSIONS,
];
export const DOCUMENT_FILE_SIZE_LIMIT = 1024 * 1024 * 10; // 10 MB
