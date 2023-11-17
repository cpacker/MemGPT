import { kv } from "@vercel/kv";
import { NextRequest, NextResponse } from "next/server";
import { nanoid } from "nanoid";
import { Bot } from "@/app/store/bot";

const DAYS_TO_LIVE = 30;
const TTL = 60 * 60 * 24 * DAYS_TO_LIVE;
const MAX_KEY_GENERATION_RETRY = 100;

export interface ShareResponse {
  key: string;
  url: string;
}

async function getKey() {
  let key;
  let counter = 0;

  do {
    key = nanoid();
    counter++;
  } while ((await kv.exists(key)) && counter < MAX_KEY_GENERATION_RETRY);

  if (counter === MAX_KEY_GENERATION_RETRY) {
    // Handle the case when a unique key was not found within the maximum allowed iterations
    throw new Error("Failed to generate a unique key");
  }
  return key;
}

export async function POST(req: NextRequest) {
  try {
    const body: { bot: Bot } = await req.json();

    const key = await getKey();
    body.bot.share = { ...body.bot.share, id: key };
    const data = await kv.set<{ bot: Bot }>(key, body, {
      ex: TTL,
    });
    if (!data) {
      throw new Error(`Can't store bot with key ${key}`);
    }

    const protocol = req.headers.get("x-forwarded-proto") || "http";
    const url = `${protocol}://${req.headers.get("host")}/b/${key}`;

    console.log(`[Share] shared bot '${body.bot.name}' created at ${url}`);

    return NextResponse.json({
      key: key,
      url: url,
      data: data,
      days: DAYS_TO_LIVE,
    } as ShareResponse);
  } catch (error) {
    console.error("[Share] error while sharing bot", error);
    return NextResponse.json(
      {
        error: true,
        msg: (error as Error).message,
      },
      {
        status: 500,
      },
    );
  }
}

export const runtime = "edge";
