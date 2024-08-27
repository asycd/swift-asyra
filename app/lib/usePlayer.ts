import { useRef, useState } from "react";

export function usePlayer() {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioContext = useRef<AudioContext | null>(null);
  const source = useRef<AudioBufferSourceNode | null>(null);

  async function play(stream: ReadableStream, callback: () => void) {
    stop(); // Stop any existing playback
    audioContext.current = new AudioContext({ sampleRate: 24000 });

    const reader = stream.getReader();
    const chunks = [];

    try {
      setIsPlaying(true);
      
      let result = await reader.read();
      while (!result.done) {
        chunks.push(...result.value);  // Collect all chunks of data
        result = await reader.read();
      }

      const buffer = new Float32Array(chunks);
      const audioBuffer = audioContext.current.createBuffer(
        1,
        buffer.length,
        audioContext.current.sampleRate
      );
      audioBuffer.copyToChannel(buffer, 0);

      source.current = audioContext.current.createBufferSource();
      source.current.buffer = audioBuffer;
      source.current.connect(audioContext.current.destination);
      
      source.current.start();
      source.current.onended = () => {
        stop();
        callback();
      };
    } catch (error) {
      console.error("Error during audio playback:", error);
      stop();
    }
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
