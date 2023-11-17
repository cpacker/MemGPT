import { URLDetailContent } from "./url";
import { FileWrap } from "../../utils/file";

export async function getDetailContentFromFile(
  file: FileWrap,
): Promise<URLDetailContent> {
  switch (file.extension) {
    case "pdf": {
      return await getPDFFileDetail(file);
    }
    case "txt": {
      return await getTextFileDetail(file);
    }
    case "jpg":
    case "jpeg": {
      return await getImageFileDetail(file);
    }
    default: {
      throw new Error("Not supported file type");
    }
  }
}

async function getPDFFileDetail(file: FileWrap): Promise<URLDetailContent> {
  const fileDataUrl = await file.readData({ asURL: true });
  const pdfBase64 = fileDataUrl.split(",")[1];

  const response = await fetch("/api/fetch", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      pdf: pdfBase64,
      fileName: file.name,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data as URLDetailContent;
}

async function getTextFileDetail(file: FileWrap): Promise<URLDetailContent> {
  const textContent = await file.readData();
  const response = await fetch("/api/fetch", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: textContent,
      fileName: file.name,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data as URLDetailContent;
}

async function getImageFileDetail(file: FileWrap) {
  const response = await fetch(`/api/upload?filename=${file.name}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: file.file,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  console.log(data);
  return data as URLDetailContent;
}
