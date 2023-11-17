export type Embedding = {
  text: string;
  embedding: number[];
};

export type URLDetail = {
  url: string;
  size: number;
  type: "text/html" | "application/pdf" | "text/plain" | "image/jpeg";
  embeddings?: Embedding[];
};

export type URLDetailContent = URLDetail & {
  content?: string;
};

export const isURL = (text: string) => {
  const isUrlRegex = /^(https?|ftp):\/\/[^\s/$.?#].[^\s]*$/i;
  return isUrlRegex.test(text);
};

export async function fetchSiteContent(
  site: string,
): Promise<URLDetailContent> {
  const response = await fetch(`/api/fetch?site=${site}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data as URLDetailContent;
}
