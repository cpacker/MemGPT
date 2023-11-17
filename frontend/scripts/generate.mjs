import {
  serviceContextFromDefaults,
  storageContextFromDefaults,
  SimpleDirectoryReader,
  VectorStoreIndex,
} from "llamaindex";

import {
  DATASOURCES_CACHE_DIR,
  DATASOURCES_DIR,
  DATASOURCES_CHUNK_SIZE,
  DATASOURCES_CHUNK_OVERLAP,
} from "./constants.mjs";
import { exit } from "process";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";

async function getRuntime(func) {
  const start = Date.now();
  await func();
  const end = Date.now();
  return end - start;
}

async function generateDatasource(serviceContext, datasource) {
  console.log(`Generating storage context for datasource '${datasource}'...`);
  // Split documents, create embeddings and store them in the storage context
  const ms = await getRuntime(async () => {
    const storageContext = await storageContextFromDefaults({
      persistDir: `${DATASOURCES_CACHE_DIR}/${datasource}`,
    });
    const documents = await new SimpleDirectoryReader().loadData({
      directoryPath: `${DATASOURCES_DIR}/${datasource}`,
    });
    await VectorStoreIndex.fromDocuments(documents, {
      storageContext,
      serviceContext,
    });
  });
  console.log(
    `Storage context for datasource '${datasource}' successfully generated in ${
      ms / 1000
    }s.`,
  );
}

async function ensureEnv(fileName) {
  try {
    const __dirname = path.dirname(new URL(import.meta.url).pathname);
    const envFileContent = await fs.promises.readFile(
      path.join(__dirname, "..", fileName),
    );
    const envConfig = dotenv.parse(envFileContent);
    if (envConfig && envConfig.OPENAI_API_KEY) {
      process.env.OPENAI_API_KEY = envConfig.OPENAI_API_KEY;
    } else {
      throw new Error(`OPENAI_API_KEY not found in '${fileName}'`);
    }
  } catch (e) {
    console.log(`Error getting OPENAI_API_KEY from ${fileName}: ${e.message}`);
    exit(1);
  }
  console.log(`Using OPENAI_API_KEY=${process.env.OPENAI_API_KEY}`);
}

const datasource = process.argv[2];

if (!datasource) {
  console.log("Error: You must provide a datasource as the parameter.");
  console.log("Usage: pnpm run generate <datasource>");
  exit(1);
}

(async () => {
  // get OPENAI_API_KEY from Next.JS's .env.development.local
  await ensureEnv(".env.development.local");

  const serviceContext = serviceContextFromDefaults({
    chunkSize: DATASOURCES_CHUNK_SIZE,
    chunkOverlap: DATASOURCES_CHUNK_OVERLAP,
  });

  await generateDatasource(serviceContext, datasource);
  console.log("Finished generating datasource.");
})();
