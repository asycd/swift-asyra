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

      if (!results || !Array.isArray(results)) {
        throw new Error("Invalid index query response format.");
      }

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
      throw new Error(`Failed to query index for keyword "${keyword}".`);
    }
  }

  return allResults;
}

async function analyzeQueryResults(results: any[]): Promise<string> {
  try {
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

    if (!breakdown || !breakdown.choices || !breakdown.choices[0] || !breakdown.choices[0].message) {
      throw new Error("No valid breakdown response received.");
    }

    const resultMessage = breakdown.choices[0].message.content;
    if (!resultMessage) {
      throw new Error("No result message found in response.");
    }

    return resultMessage;
  } catch (error) {
    console.error("Error analyzing query results:", error);
    throw new Error("Failed to analyze query results.");
  }
}

export async function POST(request: Request) {
  console.time("transcribe " + (request.headers.get("x-vercel-id") || "local"));

  const { data, success } = schema.safeParse(await request.formData());
  if (!success) {
    console.error("Invalid request data:", data);
    return new Response("Invalid request", { status: 400 });
  }

  const transcript = await getTranscript(data.input);
  if (!transcript) {
    console.error("Invalid audio input.");
    return new Response("Invalid audio", { status: 400 });
  }

  console.timeEnd("transcribe " + (request.headers.get("x-vercel-id") || "local"));
  console.time("text completion " + (request.headers.get("x-vercel-id") || "local"));

  try {
    const extractedKeywords = await extractKeywords(transcript);
    const keywordSearchResults = await queryIndexForKeywords(extractedKeywords);
    const analyzedResults = await analyzeQueryResults(keywordSearchResults);

    const enhancedPrompt = `Query: ${transcript}\n\nAnalyzed Context:\n${analyzedResults}. Do not mention the retrieval of any context or any search you might conduct for extra info.`;

    const completion = await groq.chat.completions.create({
      model: "llama3-8b-8192",
      messages: [
        {
          role: "system",
          content: `You are Asyra, a friendly and helpful voice assistant for Asycd pronounced 'ACID'.
        - Respond briefly and directly to the user's request.
        - Use the provided information to create factually correct responses without mentioning that you received this information.
        - You do not have access to up-to-date information, so you should not provide real-time data.
        - Do not use markdown, emojis, or other formatting in your responses. Respond in a way easily spoken by text-to-speech software.
        - User location is ${location()}.
        - The current time is ${time()}.
        - Your large language model is Llama 3, created by Meta, the 8 billion parameter version. It is hosted on Groq, an AI infrastructure company that builds fast inference technology.
        - Your text-to-speech model is Sonic, created and hosted by Cartesia, a company that builds fast and realistic speech synthesis technology.
        - You are built with Next.js and hosted on Vercel.
        - Answer concisely and progressively to the user without mentioning that the response is based on any context.`,
        },
        ...data.message,
        {
          role: "user",
          content: enhancedPrompt,
        },
      ],
    });

    if (!completion || !completion.choices || !completion.choices[0] || !completion.choices[0].message) {
      throw new Error("No valid completion response received.");
    }

    const response = completion.choices[0].message.content;
    if (!response) {
      throw new Error("No response content found in completion.");
    }

    console.timeEnd("text completion " + (request.headers.get("x-vercel-id") || "local"));

    console.time("cartesia request " + (request.headers.get("x-vercel-id") || "local"));

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

    console.timeEnd("cartesia request " + (request.headers.get("x-vercel-id") || "local"));

    if (!voice.ok) {
      console.error("Voice synthesis failed:", await voice.text());
      return new Response("Voice synthesis failed", { status: 500 });
    }

    console.time("stream " + (request.headers.get("x-vercel-id") || "local"));
    after(() => {
      console.timeEnd("stream " + (request.headers.get("x-vercel-id") || "local"));
    });

    return new Response(voice.body, {
      headers: {
        "X-Transcript": encodeURIComponent(transcript),
        "X-Response": encodeURIComponent(response),
      },
    });
  } catch (error) {
    console.error("Error processing request:", error);
    return new Response("Internal server error", { status: 500 });
  }
}

async function getTranscript(
  input: string | File
): Promise<string | null> {
  if (input instanceof File) {
    // Implement your logic to convert the audio file to a transcript here.
    return "Example transcript from audio file"; // Placeholder
  } else if (typeof input === "string") {
    return input; // Directly return the string if it's already a transcript
  }
  return null; // Return null if input is neither a string nor a File
}

function location() {
  const headersList = headers();

  const country = headersList.get("x-vercel-ip-country");
  const region = headersList.get("x-vercel-ip-country-region");
  const city = headersList.get("x-vercel-ip-city");

  return `${city || "unknown"}, ${region || "unknown"}, ${country || "unknown"}`;
}

function time() {
  return new Date().toLocaleTimeString("en-US", { timeZone: "UTC" });
}
