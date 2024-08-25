import Groq from "groq-sdk";
import { headers } from "next/headers";
import { z } from "zod";
import { zfd } from "zod-form-data";
import { unstable_after as after } from "next/server";
import { Index } from "@upstash/vector";
import OpenAI from 'openai';

const index = new Index({
    url: "https://optimum-sparrow-61704-us1-vector.upstash.io",
    token: "ABkFMG9wdGltdW0tc3BhcnJvdy02MTcwNC11czFhZG1pbk0yRm1OVEZsTURZdE1UVXdNUzAwTlRjMUxXRTNZak10TW1OaVpXUm1aV1U1T1RBeQ=="
});

const groq = new Groq();
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY // Make sure to set this in your environment variables
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
        input: query
    });
    return response.data[0].embedding;
}

async function queryIndex(queryVector: number[], topK: number = 5) {
    try {
        const results = await index.query({
            vector: queryVector,
            topK: topK,
            includeMetadata: true,
            includeVectors: false
        });

        return results.map(result => ({
            id: result.id,
            score: result.score,
            text: result.metadata?.text as string
        }));
    } catch (error) {
        console.error('Error querying index:', error);
        throw error;
    }
}

export async function POST(request: Request) {
    console.time("transcribe " + request.headers.get("x-vercel-id") || "local");

    const { data, success } = schema.safeParse(await request.formData());
    if (!success) return new Response("Invalid request", { status: 400 });

    const transcript = await getTranscript(data.input);
    if (!transcript) return new Response("Invalid audio", { status: 400 });

    console.timeEnd("transcribe " + request.headers.get("x-vercel-id") || "local");
    console.time("text completion " + request.headers.get("x-vercel-id") || "local");

    // Get embedding for the transcript
    const queryEmbedding = await getQueryEmbedding(transcript);

    // Query the Upstash vector index
    const queryResults = await queryIndex(queryEmbedding, 5);

    // Enhance the prompt with additional context
    const additionalContext = queryResults
        .map(result => result.text)
        .join("\n");

    const enhancedPrompt = `${transcript}\n\nAdditional Context:\n${additionalContext}`;

    const completion = await groq.chat.completions.create({
        model: "llama3-8b-8192",
        messages: [
            {
                role: "system",
                content: `- You are Asyra, a friendly and helpful voice assistant for Asycd pronounced 'ACID'
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
- You will receive context regarding about Asycd and a query. Use the context to answer concisely and progressively to the user`,
            },
            ...data.message,
            {
                role: "user",
                content: enhancedPrompt,
            },
        ],
    });

    const response = completion.choices[0].message.content;
    console.timeEnd(
        "text completion " + request.headers.get("x-vercel-id") || "local"
    );

    console.time("stream " + request.headers.get("x-vercel-id") || "local");
    after(() => {
        console.timeEnd(
            "stream " + request.headers.get("x-vercel-id") || "local"
        );
    });

    return new Response(response, {
        headers: {
            "X-Transcript": encodeURIComponent(transcript),
            "X-Response": encodeURIComponent(response),
        },
    });
}

function location() {
    const headersList = headers();

    const country = headersList.get("x-vercel-ip-country");
    const region = headersList.get("x-vercel-ip-country-region");
    const city = headersList.get("x-vercel-ip-city");

    if (!country || !region || !city) return "unknown";

    return `${city}, ${region}, ${country}`;
}

function time() {
    return new Date().toLocaleString("en-US", {
        timeZone: headers().get("x-vercel-ip-timezone") || undefined,
    });
}

async function getTranscript(input: string | File) {
    if (typeof input === "string") return input;

    try {
        const { text } = await groq.audio.transcriptions.create({
            file: input,
            model: "whisper-large-v3",
        });

        return text.trim() || null;
    } catch {
        return null; // Empty audio file
    }
}
