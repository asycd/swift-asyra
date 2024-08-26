import { useRef, useState } from "react";

export function usePlayer() {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioContext = useRef<AudioContext | null>(null);
  const source = useRef<AudioBufferSourceNode | null>(null);

  async function play(stream: ReadableStream, callback: () => void) {
    stop(); // Stop any existing playback
    audioContext.current = new AudioContext({ sampleRate: 24000 });

    const reader = stream.getReader();
    let chunks: Uint8Array[] = [];
    let result = await reader.read();

    // Read and buffer the entire stream
    while (!result.done) {
      chunks.push(result.value);
      result = await reader.read();
    }

    // Concatenate all chunks into a single Uint8Array
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const audioData = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      audioData.set(chunk, offset);
      offset += chunk.length;
    }

    // Convert Uint8Array to Float32Array
    const buffer = new Float32Array(audioData.buffer);

    // Create an AudioBuffer from the Float32Array
    const audioBuffer = audioContext.current.createBuffer(
      1,
      buffer.length,
      audioContext.current.sampleRate
    );
    audioBuffer.copyToChannel(buffer, 0);

    // Play the AudioBuffer
    source.current = audioContext.current.createBufferSource();
    source.current.buffer = audioBuffer;
    source.current.connect(audioContext.current.destination);
    source.current.onended = () => {
      stop();
      callback();
    };

    source.current.start();
    setIsPlaying(true);
  }

  function stop() {
    if (source.current) {
      source.current.stop();
      source.current.disconnect();
    }
    audioContext.current?.close();
    audioContext.current = null;
    setIsPlaying(false);
  }

  return {
    isPlaying,
    play,
    stop,
  };
}
