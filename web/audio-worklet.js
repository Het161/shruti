/* Microphone capture: Float32 -> 16kHz mono PCM16, framed for streaming STT.
 *
 * An AudioWorklet rather than the deprecated ScriptProcessorNode: worklets run on the audio
 * rendering thread, so capture is not interrupted by main-thread work like re-rendering the
 * waterfall. On a page that draws on every response, a main-thread capture path would drop
 * samples exactly when the user is most likely to be speaking.
 *
 * Frames are accumulated to ~100ms before posting. Sending every 128-sample render quantum would
 * mean ~125 WebSocket messages per second, where each message carries base64 and JSON overhead far
 * larger than its payload. 100ms is small enough that partial transcripts still feel live and
 * large enough that framing overhead stays negligible.
 */

const FRAME_SAMPLES = 1600; // 100ms at 16kHz

class PCMCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Int16Array(FRAME_SAMPLES);
    this._n = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      /* Clamp before scaling. Float32 audio can exceed [-1, 1] after gain or echo cancellation,
       * and letting that wrap in the Int16 conversion turns a loud syllable into white noise —
       * which sounds to the recogniser exactly like the user shouted gibberish. */
      const s = Math.max(-1, Math.min(1, channel[i]));
      this._buf[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;

      if (this._n === FRAME_SAMPLES) {
        /* Transfer the buffer rather than copying it, then allocate a fresh one. */
        const out = this._buf.buffer;
        this.port.postMessage(out, [out]);
        this._buf = new Int16Array(FRAME_SAMPLES);
        this._n = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PCMCapture);
