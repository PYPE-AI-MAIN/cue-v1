"""The smallest LiveKit voice agent with Cue v1 as its turn detector (see ../README.md, "Quickstart").

    pip install "cue-turn @ git+https://github.com/PYPE-AI-MAIN/cue-v1" \
        "cue-livekit @ git+https://github.com/PYPE-AI-MAIN/cue-v1#subdirectory=livekit-integration" \
        "livekit-agents[openai]~=1.8" python-dotenv
    cue serve                      # other terminal; or set CUE_URL and CUE_API_KEY to a Cue server you deployed
    python minimal_agent.py console        # keys in a .env next to this file: OPENAI_API_KEY=...
"""
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero

from cue_livekit import CueTurnDetector

load_dotenv()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    detector = CueTurnDetector()                # one per call; reads CUE_URL and CUE_API_KEY, else localhost:8090
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        turn_handling={"turn_detection": detector, "endpointing": {"min_delay": 0.3, "max_delay": 2.0}},
    )
    detector.attach(session)
    ctx.add_shutdown_callback(detector.aclose)
    await session.start(agent=Agent(instructions="You are a friendly voice assistant. Keep replies short."), room=ctx.room)
    await session.say("Hello, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
