#!/usr/bin/env python3
"""Minimal Gemini Live session driver for text or speech-to-speech chats."""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
from dataclasses import dataclass
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
        description="Start a Gemini Live session (text today, audio tomorrow)."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--response-modalities",
        default="TEXT",
        help="Comma-separated response modalities (TEXT,AUDIO).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system instruction injected into the live session.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature override.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top-p override.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional max_output_tokens override.",
    )
    parser.add_argument(
        "--log-events",
        action="store_true",
        help="Dump every LiveServerMessage for debugging.",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "speech"],
        default="text",
        help="Interaction mode. 'speech' records mic input and plays audio replies.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens live (default prints each response once it finishes).",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=4.0,
        help="Length of each recorded utterance when --mode speech is active.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for captured and generated audio in speech mode.",
    )
    parser.add_argument(
        "--mic-channels",
        type=int,
        default=1,
        help="Number of microphone channels to record (1 = mono).",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Optional prebuilt voice name for audio responses (speech mode).",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Optional PortAudio device index for microphone (speech mode).",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Optional PortAudio device index for playback (speech mode).",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Do not play synthesized audio out loud (still prints transcript).",
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
        "max_output_tokens": args.max_output_tokens,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    config_kwargs: dict = {"response_modalities": response_modalities}
    if gen_kwargs:
        config_kwargs["generation_config"] = types.GenerationConfig(**gen_kwargs)
    if args.system_prompt:
        config_kwargs["system_instruction"] = args.system_prompt
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
    """Keeps CLI output tidy while streaming chunks."""

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
) -> None:
    if stream and not printer:
        printer = TurnPrinter()
    if stream and printer:
        printer.start()
    buffer: List[str] = []
    buffered_audio: List[Tuple[str, bytes]] = []
    audio_seen = False
    async for server_message in session.receive():
        chunks = extract_text_chunks(server_message)
        if stream and printer:
            printer.feed(chunks)
        else:
            buffer.extend(chunks)
        audio_chunks = extract_audio_chunks(server_message)
        if audio_player and audio_chunks:
            audio_seen = True
            if stream:
                for mime, data in audio_chunks:
                    await audio_player.play(mime, data)
            else:
                buffered_audio.extend(audio_chunks)
        elif audio_chunks:
            audio_seen = True
        log_misc_events(server_message, verbose=verbose)
    if stream and printer:
        printer.finish()
    else:
        text = "".join(buffer).strip()
        if text:
            print("model>", text)
        elif audio_seen:
            print("model> [audio response]")
    if audio_player and buffered_audio:
        for mime, data in buffered_audio:
            await audio_player.play(mime, data)


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

    async def play(self, mime_type: str, data: bytes) -> None:
        if not self.enabled or not data:
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
            data,
            sample_rate,
            channels,
        )

    def _play_blocking(self, data: bytes, sample_rate: int, channels: int) -> None:
        if not data:
            return
        audio = self.np.frombuffer(data, dtype=self.np.int16)
        if channels > 1 and len(audio) % channels == 0:
            audio = audio.reshape(-1, channels)
        self.sd.play(audio, samplerate=sample_rate, device=self.device)
        self.sd.wait()


async def record_microphone(
    duration_seconds: float,
    samplerate: int,
    channels: int,
    device: int | None = None,
) -> bytes:
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "sounddevice and numpy are required for speech mode. "
            "Install them with 'pip install sounddevice numpy'."
        ) from exc

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")

    frames = int(duration_seconds * samplerate)

    def _record() -> bytes:
        recording = sd.rec(
            frames,
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            device=device,
        )
        sd.wait()
        arr = np.array(recording, copy=True)
        return arr.tobytes()

    return await asyncio.to_thread(_record)


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
    verbose: bool = False,
    stream: bool = False,
    clip_seconds: float,
    sample_rate: int,
    mic_channels: int,
    playback_enabled: bool,
    input_device: int | None,
    output_device: int | None,
) -> None:
    printer = TurnPrinter() if stream else None
    player = AudioPlayer(
        preferred_rate=sample_rate,
        preferred_channels=mic_channels,
        enabled=playback_enabled,
        device=output_device,
    )
    print(
        "system> Speech mode. Press Enter to record audio, "
        "or type text manually. Use :q or Ctrl+C to exit."
    )
    async with client.aio.live.connect(model=model, config=config) as session:
        while True:
            try:
                user_input = await ainput("you (Enter=record)> ")
            except (EOFError, KeyboardInterrupt):
                print("\nsystem> Exiting.")
                return
            lowered = user_input.strip().lower()
            if lowered in {":q", ":quit", ":exit"}:
                print("system> Bye!")
                return
            if user_input.strip():
                await session.send_client_content(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=user_input.strip())],
                        )
                    ],
                    turn_complete=True,
                )
                await handle_responses(
                    session,
                    stream=stream,
                    printer=printer,
                    verbose=verbose,
                    audio_player=player,
                )
                continue
            try:
                print(f"system> Recording for {clip_seconds:.1f} s…")
                audio_bytes = await record_microphone(
                    duration_seconds=clip_seconds,
                    samplerate=sample_rate,
                    channels=mic_channels,
                    device=input_device,
                )
            except Exception as exc:  # pragma: no cover - runtime feedback
                print(f"system> Recording failed: {exc}")
                continue
            if not audio_bytes:
                print("system> No audio captured, try again.")
                continue
            blob = types.Blob(
                data=audio_bytes,
                mime_type=f"audio/pcm;rate={sample_rate}",
            )
            await session.send_realtime_input(audio=blob)
            await session.send_realtime_input(audio_stream_end=True)
            await handle_responses(
                session,
                stream=stream,
                printer=printer,
                verbose=verbose,
                audio_player=player,
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
            clip_seconds=args.clip_seconds,
            sample_rate=args.sample_rate,
            mic_channels=args.mic_channels,
            playback_enabled=not args.no_playback,
            input_device=args.input_device,
            output_device=args.output_device,
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
