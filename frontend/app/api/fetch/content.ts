import { unified } from "unified";
import parse from "rehype-parse";
import rehype2remark from "rehype-remark";
import stringify from "remark-stringify";
import axios from "axios";
import pdf from "pdf-parse";
import { remove } from "unist-util-remove";
import { URLDetailContent } from "@/app/client/fetch/url";

function removeCommentsAndTables() {
  return (tree: any) => {
    remove(tree, { type: "comment" });
    remove(tree, { tagName: "table" });
  };
}

async function htmlToMarkdown(html: string): Promise<string> {
  const processor = unified()
    .use(parse) // Parse the HTML
    .use(removeCommentsAndTables) // Remove comment nodes
    .use(rehype2remark as any) // Convert it to Markdown
    .use(stringify); // Stringify the Markdown

  const file = await processor.process(html);
  return String(file);
}

export async function fetchContentFromURL(
  url: string,
): Promise<URLDetailContent> {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error("Failure fetching content from provided URL");
  }

  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("text/html")) {
    const htmlContent = await response.text();
    const markdownContent = await htmlToMarkdown(htmlContent);
    return {
      url,
      content: markdownContent,
      size: htmlContent.length,
      type: "text/html",
    };
  }

  if (contentType.includes("application/pdf")) {
    const response = await axios.get(url, {
      responseType: "arraybuffer",
    });
    const pdfBuffer = response.data;
    const pdfData = await pdf(pdfBuffer);
    const result = {
      url,
      content: pdfData.text,
      size: pdfData.text.length,
      type: "application/pdf",
    } as URLDetailContent;
    return result;
  }

  throw new Error("URL provided is not a PDF or HTML document");
}

export const getPDFContentFromBuffer = async (pdfBuffer: Buffer) => {
  const data = await pdf(pdfBuffer);
  const content = data.text;
  const size = data.text.length;

  return {
    content,
    size,
    type: "application/pdf",
  };
};
