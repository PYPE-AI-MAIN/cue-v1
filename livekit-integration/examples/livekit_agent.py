"""A LiveKit voice agent in which Cue v1 decides when the caller has finished (see ../README.md).

From the repo root:

    pip install -e . -e ./livekit-integration "livekit-agents[openai]~=1.8"
    cue serve                                   # in another terminal: downloads and serves Cue v1
    OPENAI_API_KEY=... python livekit-integration/examples/livekit_agent.py console

Settings (environment):
    CUE_URL          the Cue server (default http://localhost:8090/v1)
    CUE_SMART_TURN   1: also Smart Turn v3.2 on the caller's audio (p(done) x sqrt(Smart Turn), as benchmarked)
    CUE_BARGE_IN     1: Cue decides whether the caller's words over her stop her (LiveKit's own interruptions off)

The STT, LLM and TTS are OpenAI's here so one key runs the example; any LiveKit plugins work in their place.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]                       # run from a clone: cue_livekit, and cue at the root
sys.path[:0] = [str(HERE), str(HERE.parent)]

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli  # noqa: E402
from livekit.plugins import openai, silero  # noqa: E402

from cue_livekit import CueTurnDetector  # noqa: E402

CUE_URL = os.getenv("CUE_URL", "http://localhost:8090/v1")
SMART_TURN = os.getenv("CUE_SMART_TURN") == "1"
BARGE_IN = os.getenv("CUE_BARGE_IN") == "1"
logging.getLogger("cue.livekit").setLevel(logging.INFO)          # one line per decision


def make_detector(smart_turn_model=None) -> CueTurnDetector:
    """One per call: it holds the call's state (and its own Smart Turn audio buffer)."""
    smart_turn = smart_turn_model.fresh() if smart_turn_model is not None else None
    return CueTurnDetector(CUE_URL, smart_turn=smart_turn, barge_in=BARGE_IN,
                           threshold=0.3 if smart_turn else 0.5)


def turn_handling(detector: CueTurnDetector) -> dict:
    th = {
        "turn_detection": detector,
        # Cue says how likely the caller is done; LiveKit waits min_delay when likely, max_delay when not
        "endpointing": {"min_delay": 0.3, "max_delay": 2.0},
    }
    if BARGE_IN:
        # Cue decides barge-in, so LiveKit's own VAD interruptions are off, and the STT keeps hearing the
        # caller while she speaks (otherwise LiveKit sends it silence during a speech it may not interrupt)
        th["interruption"] = {"enabled": False, "discard_audio_if_uninterruptible": False}
    return th


class Assistant(Agent):
    def __init__(self, detector: CueTurnDetector) -> None:
        super().__init__(instructions="You are a friendly receptionist for a clinic. Keep replies short.")
        self.cue = detector

    async def stt_node(self, audio, model_settings):
        # the caller's audio passes through the detector on its way to the STT (Smart Turn listens to it)
        async for event in Agent.default.stt_node(self, self.cue.listen_audio(audio), model_settings):
            yield event

    async def tts_node(self, text, model_settings):
        # her words pass through the detector on their way to the TTS (barge-in reads what she has said)
        async for frame in Agent.default.tts_node(self, self.cue.listen_text(text), model_settings):
            yield frame


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    if SMART_TURN:
        from cue_livekit import SmartTurn
        proc.userdata["smart_turn"] = SmartTurn()                # the model, once per process (downloads it once)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    detector = make_detector(ctx.proc.userdata.get("smart_turn"))
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        turn_handling=turn_handling(detector),
    )
    detector.attach(session)
    ctx.add_shutdown_callback(detector.aclose)
    await session.start(agent=Assistant(detector), room=ctx.room)
    await session.say("Hello, this is the clinic. How can I help you?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
