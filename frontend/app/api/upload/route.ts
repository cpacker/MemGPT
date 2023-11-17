import { put } from "@vercel/blob";
import { NextResponse } from "next/server";
import { URLDetail } from "../../client/fetch/url";

export async function POST(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const filename = searchParams.get("filename");
  if (!filename || !request.body) {
    return NextResponse.json(
      { error: "Missing filename URL parameter or request body" },
      { status: 400 },
    );
  }

  try {
    const blob = await put(filename, request.body, {
      access: "public",
    });

    const json = {
      type: blob.contentType as URLDetail["type"],
      url: blob.url,
      // TODO: needs to return the size of the uploaded file
      size: NaN,
    };

    return NextResponse.json<URLDetail>(json);
  } catch (error) {
    console.error("[Upload]", error);
    return NextResponse.json(
      {
        error: (error as Error).message,
      },
      {
        status: 500,
      },
    );
  }
}
