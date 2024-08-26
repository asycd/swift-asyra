import Groq from "groq-sdk";
import { headers } from "next/headers";
import { z } from "zod";
import { zfd } from "zod-form-data";
import { unstable_after as after } from "next/server";
import { Index } from "@upstash/vector";
import OpenAI from 'openai';

// **Recommendation: Use environment variables for sensitive information**
const index = new Index({
    url: process.env.UPSTASH_VECTOR_URL!, // e.g., "https://your-vector.upstash.io"
    token: process.env.UPSTASH_VECTOR_TOKEN!, // Your Upstash vector token
});

const groq = new Groq();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

const schema = zfd.formData({
  input: z.union([zfd.text(), zfd.file()]),
  message: zfd.repeatableOfType(
    zfd.json(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string(),
      })
    )
  ),
});

async function getQueryEmbedding(query: string): Promise<number[]> {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: query,
    });
    if (!response.data || !response.data[0] || !response.data[0].embedding) {
      throw new Error("Invalid embedding response format.");
    }
    return response.data[0].embedding;
  } catch (error) {
    console.error("Error generating query embedding:", error);
    throw new Error("Failed to generate query embedding.");
  }
}

async function extractKeywords(transcript: string): Promise<string[]> {
  try {
    const keywordExtractionResponse = await groq.chat.completions.create({
      model: "llama3-8b-8192",
      messages: [
        {
          role: "system",
          content: `You are an AI model specialized in keyword extraction. Given a transcript, extract the maximum 5 most relevant and specific keywords from the transcript. No need to use double or single quotes. Return the keywords as a comma-separated string.`,
        },
        {
          role: "user",
          content: `Transcript: ${transcript}`,
        },
      ],
    });

    if (!keywordExtractionResponse || !keywordExtractionResponse.choices || !keywordExtractionResponse.choices[0] || !keywordExtractionResponse.choices[0].message) {
      throw new Error("No valid keyword extraction response received.");
    }

    const keywordMessage = keywordExtractionResponse.choices[0].message.content;
    if (!keywordMessage) {
      throw new Error("No content in keyword extraction response.");
    }

    const cleanedKeywordMessage = keywordMessage.trim();
    const keywords = cleanedKeywordMessage.split(',').map((keyword) => keyword.trim());

    if (Array.isArray(keywords)) {
      return keywords.slice(0, 5);
    } else {
      throw new Error("Unexpected format for keyword extraction output.");
    }
  } catch (error) {
    console.error("Error extracting keywords:", error);
    throw new Error("Keyword extraction failed.");
  }
}

async function queryIndexForKeywords(keywords: string[], topK: number = 5) {
  const allResults = [];

  for (const keyword of keywords) {
    try {
      const keywordEmbedding = await getQueryEmbedding(keyword);

      const results = await index.query({
        vector: keywordEmbedding,
        topK: topK,
        includeMetadata: true,
        includeVectors: false,
      });

      if (!results ||
