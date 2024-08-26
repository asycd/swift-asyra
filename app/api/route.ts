import Groq from "groq-sdk";
import { headers } from "next/headers";
import { z } from "zod";
import { zfd } from "zod-form-data";
import { unstable_after as after } from "next/server";
import { Index } from "@upstash/vector";
import OpenAI from 'openai';

// **Recommendation: Use environment variables for sensitive information**
const index = new Index({
    url: process.env.UPSTASH_VECTOR_URL, // e.g., "https://your-vector.upstash.io"
    token: process.env.UPSTASH_VECTOR_TOKEN, // Your Upstash vector token
});

const groq = new Groq();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
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
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: query,
  });
  return response.data[0].embedding;
}

async function extractKeywords(
  query: string,
  transcript: string
): Promise<string[]> {
  const keywordExtractionResponse = await groq.chat.completions.create({
    model: "llama3-8b-8192",
    messages: [
      {
        role: "system",
        content: `You are an AI model specialized in keyword extraction. Given a user's query and a related transcript, extract the maxiumum 5 most relevant and specific keywords from the transcript that align with the user's query.No need to use double or single quotes. Return the keywords as a comma-separated string e.g.User Query: "I need information about cloud computing services." keywords:cloud computing,services,information `},
      {
        role: "user",
        content: `Query: ${query}\n\nTranscript: ${transcript}`,
      },
    ],
  });

  const choices = keywordExtractionResponse.choices;
  if (!choices || !choices[0] || !choices[0].message) {
    throw new Error("No valid keyword extraction response received.");
  }

  const keywordMessage = choices[0].message.content;
  if (!keywordMessage) {
    throw new Error("No content in keyword extraction response.");
  }

  try {
    const cleanedKeywordMessage = keywordMessage.trim();

    // Use native JavaScript to split the comma-separated keywords into an array
    const keywords = cleanedKeywordMessage.split(',').map((keyword) => keyword.trim());

    if (Array.isArray(keywords)) {
      return keywords.slice(0, 5);
    } else {
      throw new Error("Unexpected format for keyword extraction output.");
    }
  } catch (error) {
    console.error("Error parsing keywords:", error);
    throw error;
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

      allResults.push({
        keyword,
        results: results.map((result) => ({
          id: result.id,
          score: result.score,
          text: result.metadata?.text as string,
        })),
      });
    } catch (error) {
      console.error(`Error querying index for keyword "${keyword}":`, error);
      throw error;
    }
  }

  return allResults;
}

async function analyzeQueryResults(results: any[]): Promise<string> {
  const breakdown = await groq.chat.completions.create({
    model: "llama3-8b-8192",
    messages: [
      {
        role: "system",
        content: `You are a specialized assistant tasked with breaking down query results into digestible insights. Analyze the following results and provide a structured summary.`,
      },
      {
        role: "user",
        content: JSON.stringify(results),
      },
    ],
  });

  const resultMessage = breakdown.choices?.[0]?.message?.content;
  if (!resultMessage) {
    throw new Error("No result message found in response.");
  }

  return resultMessage;
}

export async function POST(request: Request) {
  console.time("transcribe " + request.headers.get("x-vercel-id") || "local");

  const { data, success } = schema.safeParse(await request.formData());
  if (!success) return new Response("Invalid request", { status: 400 });

  const transcript = await getTranscript(data.input);
  if (!transcript) return new Response("Invalid audio", { status: 400 });

  console.timeEnd("transcribe " + request.headers.get("x-vercel-id") || "local");
  console.time("text completion " + request.headers.get("x-vercel-id") || "local");

  const extractedKeywords = await extractKeywords(data.message[0].content, transcript);

  const keywordSearchResults = await queryIndexForKeywords(extractedKeywords);

  const analyzedResults = await analyzeQueryResults(keywordSearchResults);

  const enhancedPrompt = `${transcript}\n\nAnalyzed Context:\n${analyzedResults}`;

  const completion = await groq.chat.completions.create({
    model: "llama3-8b-8192",
    messages: [
      {
        role: "system",
        content: `- You are Asyra, a friendly and helpful voice assistant for Asycd pronounced 'ACID'.
            - Respond briefly to the user's request, and do not provide unnecessary information.
            - Use the information provided to create factually correct responses well measured.
            - You do not have access to up-to-date information, so you should not provide real-time data.
            - You are not capable of performing actions other than responding to the user.
            - Do not use markdown, emojis, or other formatting in your responses. Respond in a way easily spoken by text-to-speech software.
            - User location is ${location()}.
            - The current time is ${time()}.
            - Your large language model is Llama 3, created by Meta, the 8 billion parameter version. It is hosted on Groq, an AI infrastructure company that builds fast inference technology.
            - Your text-to-speech model is Sonic, created and hosted by Cartesia, a company that builds fast and realistic speech synthesis technology.
            - You are built with Next.js and hosted on Vercel.
            - You will receive context regarding about Asycd and a query. Use the context to answer concisely and progressively to the user.`,
      },
      ...data.message,
      {
        role: "user",
        content: enhancedPrompt,
      },
    ],
  });

  const response = completion.choices?.[0]?.message?.content;
  if (!response) {
    throw new Error("No response content found in completion.");
  }

  console.timeEnd("text completion " + request.headers.get("x-vercel-id") || "local");

  console.time("cartesia request " + request.headers.get("x-vercel-id") || "local");

  const voice = await fetch("https://api.cartesia.ai/tts/bytes", {
    method: "POST",
    headers: {
      "Cartesia-Version": "2024-06-30",
      "Content-Type": "application/json",
      "X-API-Key": process.env.CARTESIA_API_KEY!,
    },
    body: JSON.stringify({
      model_id: "sonic-english",
      transcript: response,
      voice: {
        mode: "id",
        id: "79a125e8-cd45-4c13-8a67-188112f4dd22",
      },
      output_format: {
        container: "raw",
        encoding: "pcm_f32le",
        sample_rate: 24000,
      },
    }),
  });

  console.timeEnd("cartesia request " + request.headers.get("x-vercel-id") || "local");

  if (!voice.ok) {
    console.error(await voice.text());
    return new Response("Voice synthesis failed", { status: 500 });
  }

  console.time("stream " + request.headers.get("x-vercel-id") || "local");
  after(() => {
    console.timeEnd("stream " + request.headers.get("x-vercel-id") || "local");
  });

  return new Response(voice.body, {
    headers: {
      "X-Transcript": encodeURIComponent(transcript),
      "X-Response": encodeURIComponent(response),
    },
  });
}

async function getTranscript(
  input: string | File
): Promise<string | null> {
  if (input instanceof File) {
    return "Example transcript from audio file";
  } else if (typeof input === "string") {
    return input;
  }
  return null;
}

function location() {
  const headersList = headers();

  const country = headersList.get("x-vercel-ip-country");
  const region = headersList.get("x-vercel-ip-country-region");
  const city = headersList.get("x-vercel-ip-city");

  return `${city}, ${region}, ${country}`;
}

function time() {
  return new Date().toLocaleTimeString("en-US", { timeZone: "UTC" });
}
