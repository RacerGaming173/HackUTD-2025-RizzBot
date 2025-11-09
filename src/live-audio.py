#!/usr/bin/env python3
"""Gemini Live CLI for text or speech (open-mic) interactions."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "google-genai is required. Install it with 'pip install google-genai'."
    ) from exc


DEFAULT_MODEL = "models/gemini-2.0-flash-exp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a Gemini Live session (text or speech)."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--response-modalities",
        default="TEXT",
        help="Comma-separated response modalities (TEXT,AUDIO).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System instruction inserted into the session.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="File containing the system prompt (overrides --system-prompt).",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--log-events",
        action="store_true",
        help="Dump every LiveServerMessage for debugging.",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "speech"],
        default="text",
        help="text mode (type/see replies) or speech (open mic).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream text tokens as they arrive (default prints once per turn).",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=0.3,
        help="Microphone chunk size in seconds for speech mode.",
    )
    parser.add_argument(
        "--playback-chunk-seconds",
        type=float,
        default=0.6,
        help="Buffered audio duration (seconds) before playback starts.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate.",
    )
    parser.add_argument(
        "--mic-channels",
        type=int,
        default=1,
        help="Microphone channel count (1 = mono).",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Name of a prebuilt Gemini voice (speech responses).",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="PortAudio device index for the microphone.",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="PortAudio device index for playback.",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Do not play audio responses (speech mode still prints transcripts).",
    )
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv()
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )
    if not key:
        raise SystemExit(
            "Missing GEMINI_API_KEY/GOOGLE_API_KEY environment variable."
        )
    return key


def resolve_system_prompt(args: argparse.Namespace) -> str | None:
    if args.system_prompt_file:
        path = Path(args.system_prompt_file)
        if not path.is_file():
            raise SystemExit(f"System prompt file not found: {path}")
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"Failed to read system prompt file: {exc}") from exc
    return args.system_prompt


def build_config(args: argparse.Namespace) -> types.LiveConnectConfig:
    wants_audio_out = args.mode == "speech"
    response_modalities = [
        modality.strip().upper()
        for modality in args.response_modalities.split(",")
        if modality.strip()
    ] or ["TEXT"]
    if wants_audio_out:
        response_modalities = ["AUDIO"]

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_output_tokens": args.max_output_tokens,
    }
    gen_kwargs = {key: val for key, val in gen_kwargs.items() if val is not None}

    config_kwargs: dict = {"response_modalities": response_modalities}
    if gen_kwargs:
        config_kwargs["generation_config"] = types.GenerationConfig(**gen_kwargs)

    system_prompt = resolve_system_prompt(args)
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    if args.voice:
        config_kwargs["speech_config"] = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=args.voice
                )
            )
        )

    return types.LiveConnectConfig(**config_kwargs)


async def ainput(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


@dataclass
class TurnPrinter:
    """Simple helper to stream text tokens with a clean prefix."""

    mid_turn: bool = False

    def start(self) -> None:
        self.mid_turn = False

    def feed(self, chunks: Iterable[str]) -> None:
        for chunk in chunks:
            if not chunk:
                continue
            if not self.mid_turn:
                print("model> ", end="", flush=True)
                self.mid_turn = True
            print(chunk, end="", flush=True)

    def finish(self) -> None:
        if self.mid_turn:
            print()
        self.mid_turn = False


def extract_text_chunks(message: types.LiveServerMessage) -> List[str]:
    chunks: List[str] = []
    content = message.server_content
    if content and content.model_turn:
        for part in content.model_turn.parts or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return chunks


def extract_audio_chunks(
    message: types.LiveServerMessage,
) -> List[Tuple[str, bytes]]:
    results: List[Tuple[str, bytes]] = []
    content = message.server_content
    if not content or not content.model_turn:
        return results
    for part in content.model_turn.parts or []:
        blob = getattr(part, "inline_data", None)
        if not blob or not getattr(blob, "mime_type", ""):
            continue
        mime = blob.mime_type or ""
        if not mime.startswith("audio/"):
            continue
        data = blob.data
        if isinstance(data, str):
            try:
                decoded = base64.b64decode(data)
            except Exception:
                continue
        else:
            decoded = data or b""
        if decoded:
            results.append((mime, decoded))
    return results


def log_misc_events(
    message: types.LiveServerMessage, verbose: bool = False
) -> None:
    if message.setup_complete:
        print("system> Session setup complete.")
    if message.tool_call:
        print("system> Tool call requested:", message.tool_call)
    if message.tool_call_cancellation:
        print("system> Tool call cancelled:", message.tool_call_cancellation)
    if message.go_away:
        reason = getattr(message.go_away, "reason", "unspecified")
        print("system> Server requested disconnect:", reason)
    if verbose:
        print("debug>", message.model_dump())


async def handle_responses(
    session: genai.live.AsyncSession,
    *,
    stream: bool,
    printer: TurnPrinter | None,
    verbose: bool = False,
    audio_player: "AudioPlayer | None" = None,
    audio_chunk_seconds: float | None = None,
    continuous: bool = False,
) -> None:
    if stream and not printer:
        printer = TurnPrinter()

    while True:
        if stream and printer:
            printer.start()

        text_buffer: List[str] = []
        audio_buffer = bytearray()
        audio_mime: str | None = None
        bytes_per_frame = 2
        chunk_threshold_bytes: int | None = None
        audio_seen = False
        turn_completed = False

        async def flush_ready() -> None:
            nonlocal audio_buffer, audio_mime, chunk_threshold_bytes, bytes_per_frame
            if not audio_buffer or not audio_mime:
                audio_buffer = bytearray()
                audio_mime = None
                chunk_threshold_bytes = None
                bytes_per_frame = 2
                return
            if audio_player:
                await audio_player.play_buffer(audio_mime, [bytes(audio_buffer)])
            audio_buffer = bytearray()
            audio_mime = None
            chunk_threshold_bytes = None
            bytes_per_frame = 2

        async def flush_threshold() -> None:
            if (
                not audio_player
                or chunk_threshold_bytes is None
                or not audio_mime
            ):
                return
            while len(audio_buffer) >= chunk_threshold_bytes:
                chunk = bytes(audio_buffer[:chunk_threshold_bytes])
                del audio_buffer[:chunk_threshold_bytes]
                await audio_player.play_buffer(audio_mime, [chunk])

        def should_end_turn(message: types.LiveServerMessage) -> bool:
            content = message.server_content
            if not content:
                return False
            return bool(
                content.turn_complete
                or content.generation_complete
                or content.waiting_for_input
            )

        async for server_message in session.receive():
            text_chunks = extract_text_chunks(server_message)
            if stream and printer:
                printer.feed(text_chunks)
            else:
                text_buffer.extend(text_chunks)

            audio_chunks = extract_audio_chunks(server_message)
            if audio_chunks:
                audio_seen = True
                for mime, data in audio_chunks:
                    if audio_mime and mime != audio_mime:
                        await flush_ready()
                    if not audio_mime:
                        audio_mime = mime
                        _, rate, channels = parse_audio_metadata(
                            mime, 16000, 1
                        )
                        bytes_per_frame = max(1, channels) * 2
                        if (
                            audio_player
                            and audio_chunk_seconds
                            and audio_chunk_seconds > 0
                        ):
                            window = max(0.05, audio_chunk_seconds)
                            chunk_threshold_bytes = max(
                                bytes_per_frame,
                                int(rate * window * bytes_per_frame),
                            )
                        else:
                            chunk_threshold_bytes = None
                    audio_buffer.extend(data)
                    await flush_threshold()

            log_misc_events(server_message, verbose=verbose)
            if should_end_turn(server_message):
                await flush_ready()
                turn_completed = True
                break

        if not turn_completed:
            await flush_ready()
            return

        if stream and printer:
            printer.finish()
        else:
            text = "".join(text_buffer).strip()
            if text:
                print("model>", text)
            elif audio_seen:
                print("model> [audio response]")

        if audio_buffer:
            await flush_ready()

        if not continuous:
            return


def parse_audio_metadata(
    mime_type: str, fallback_rate: int, fallback_channels: int
) -> Tuple[str, int, int]:
    base = mime_type.split(";", 1)[0].strip().lower()
    sample_rate = fallback_rate
    channels = fallback_channels
    for token in mime_type.split(";")[1:]:
        token = token.strip().lower()
        if token.startswith("rate="):
            try:
                sample_rate = int(token.split("=", 1)[1])
            except ValueError:
                pass
        elif token.startswith("channels="):
            try:
                channels = int(token.split("=", 1)[1])
            except ValueError:
                pass
    return base, sample_rate, max(1, channels)


class AudioPlayer:
    def __init__(
        self,
        preferred_rate: int,
        preferred_channels: int = 1,
        enabled: bool = True,
        device: int | None = None,
    ) -> None:
        self.enabled = enabled
        if not enabled:
            self.sd = None
            self.np = None
            return
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as exc:  # pragma: no cover - import guard
            raise SystemExit(
                "sounddevice and numpy are required for speech mode. "
                "Install them with 'pip install sounddevice numpy'."
            ) from exc
        self.sd = sd
        self.np = np
        self.preferred_rate = preferred_rate
        self.preferred_channels = preferred_channels
        self.device = device

    async def play_buffer(
        self, mime_type: str, chunks: Iterable[bytes]
    ) -> None:
        if not self.enabled:
            return
        payload = b"".join(chunks)
        if not payload:
            return
        base, sample_rate, channels = parse_audio_metadata(
            mime_type, self.preferred_rate, self.preferred_channels
        )
        if base not in {"audio/pcm", "audio/raw"}:
            print(
                f"system> Received audio '{mime_type}', "
                "but only raw PCM playback is supported."
            )
            return
        await asyncio.to_thread(
            self._play_blocking,
            payload,
            sample_rate,
            channels,
        )

    def _play_blocking(
        self, data: bytes, sample_rate: int, channels: int
    ) -> None:
        if not data:
            return
        audio = self.np.frombuffer(data, dtype=self.np.int16)
        if channels > 1 and len(audio) % channels == 0:
            audio = audio.reshape(-1, channels)
        self.sd.play(audio, samplerate=sample_rate, device=self.device)
        self.sd.wait()


async def stream_microphone(
    session: genai.live.AsyncSession,
    *,
    samplerate: int,
    channels: int,
    chunk_seconds: float,
    device: int | None,
    stop_event: asyncio.Event,
) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "sounddevice and numpy are required for speech mode. "
            "Install them with 'pip install sounddevice numpy'."
        ) from exc

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def callback(indata, frames, time_info, status):  # pragma: no cover
        if status:
            print(f"mic> {status}")
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    block_frames = max(1, int(samplerate * chunk_seconds))
    try:
        with sd.RawInputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            blocksize=block_frames,
            callback=callback,
            device=device,
        ):
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if not chunk:
                    continue
                blob = types.Blob(
                    data=chunk,
                    mime_type=f"audio/pcm;rate={samplerate}",
                )
                await session.send_realtime_input(audio=blob)
    finally:
        with contextlib.suppress(Exception):
            await session.send_realtime_input(audio_stream_end=True)


async def text_live_loop(
    client: genai.Client,
    model: str,
    config: types.LiveConnectConfig,
    verbose: bool = False,
    stream: bool = False,
) -> None:
    printer = TurnPrinter() if stream else None
    print("system> Connecting to Gemini Live… (Ctrl+C or :q to exit)")
    async with client.aio.live.connect(model=model, config=config) as session:
        while True:
            try:
                user_text = (await ainput("you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nsystem> Exiting.")
                return
            if not user_text:
                continue
            if user_text.lower() in {":q", ":quit", ":exit"}:
                print("system> Bye!")
                return
            await session.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_text)],
                    )
                ],
                turn_complete=True,
            )
            await handle_responses(
                session,
                stream=stream,
                printer=printer,
                verbose=verbose,
            )


async def speech_live_loop(
    client: genai.Client,
    model: str,
    config: types.LiveConnectConfig,
    *,
    verbose: bool,
    stream: bool,
    chunk_seconds: float,
    sample_rate: int,
    mic_channels: int,
    playback_enabled: bool,
    input_device: int | None,
    output_device: int | None,
    playback_chunk_seconds: float,
) -> None:
    printer = TurnPrinter() if stream else None
    player = AudioPlayer(
        preferred_rate=sample_rate,
        preferred_channels=mic_channels,
        enabled=playback_enabled,
        device=output_device,
    )
    print(
        "system> Speech mode live. Start talking any time (Ctrl+C to exit). "
        "Use --clip-seconds/--playback-chunk-seconds to tune latency."
    )

    stop_event = asyncio.Event()

    async with client.aio.live.connect(model=model, config=config) as session:
        mic_task = asyncio.create_task(
            stream_microphone(
                session,
                samplerate=sample_rate,
                channels=mic_channels,
                chunk_seconds=chunk_seconds,
                device=input_device,
                stop_event=stop_event,
            )
        )
        # Future hardware buttons can call session.send_realtime_input(
        #     audio_stream_end=True
        # ) to force a model response.
        response_task = asyncio.create_task(
            handle_responses(
                session,
                stream=stream,
                printer=printer,
                verbose=verbose,
                audio_player=player,
                audio_chunk_seconds=playback_chunk_seconds,
                continuous=True,
            )
        )

        try:
            await asyncio.gather(mic_task, response_task)
        finally:
            stop_event.set()
            for task in (mic_task, response_task):
                task.cancel()
            await asyncio.gather(
                mic_task, response_task, return_exceptions=True
            )


async def main_async() -> None:
    args = parse_args()
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)
    config = build_config(args)

    if args.mode == "speech":
        await speech_live_loop(
            client=client,
            model=args.model,
            config=config,
            verbose=args.log_events,
            stream=args.stream,
            chunk_seconds=args.clip_seconds,
            sample_rate=args.sample_rate,
            mic_channels=args.mic_channels,
            playback_enabled=not args.no_playback,
            input_device=args.input_device,
            output_device=args.output_device,
            playback_chunk_seconds=args.playback_chunk_seconds,
        )
    else:
        await text_live_loop(
            client=client,
            model=args.model,
            config=config,
            verbose=args.log_events,
            stream=args.stream,
        )


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nsystem> Exiting.")


if __name__ == "__main__":
    main()
