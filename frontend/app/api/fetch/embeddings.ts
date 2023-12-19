import { Embedding } from "@/app/client/fetch/url";
import {
  DATASOURCES_CHUNK_OVERLAP,
  DATASOURCES_CHUNK_SIZE,
} from "@/scripts/constants.mjs";
import {
  Document,
  MetadataMode,
  SentenceSplitter,
  VectorStoreIndex,
  getNodesFromDocument,
  serviceContextFromDefaults,
} from "llamaindex";

export default async function splitAndEmbed(
  document: string,
): Promise<Embedding[]> {
  const nodes = getNodesFromDocument(
    new Document({ text: document }),
    new SentenceSplitter({
      chunkSize: DATASOURCES_CHUNK_SIZE,
      chunkOverlap: DATASOURCES_CHUNK_OVERLAP,
    }),
  );

  const nodesWithEmbeddings = await VectorStoreIndex.getNodeEmbeddingResults(
    nodes,
    serviceContextFromDefaults(),
    true,
  );

  return nodesWithEmbeddings.map((nodeWithEmbedding) => ({
    text: nodeWithEmbedding.getContent(MetadataMode.NONE),
    embedding: nodeWithEmbedding.getEmbedding(),
  }));
}
