import httpx
from pydantic import BaseModel, Field


class Action:
    class Valves(BaseModel):
        image_size: str = Field(default="512x512", description="Image size")

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> dict | None:
        if __event_emitter__ is None:
            return

        messages = body.get("messages", [])
        last_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    last_message = content.strip()
                    break

        if not last_message:
            await __event_emitter__(
                {"type": "status", "data": {"description": "No message to illustrate.", "done": True}}
            )
            return

        prompt = last_message[:500]

        await __event_emitter__(
            {"type": "status", "data": {"description": "Generating image...", "done": False}}
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "http://localhost:8000/v1/images/generations",
                    json={
                        "prompt": prompt,
                        "n": 1,
                        "size": self.valves.image_size,
                        "response_format": "url",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            images = []
            if isinstance(data, dict) and "data" in data:
                images = [img.get("url", "") for img in data["data"] if img.get("url")]

            if images:
                md = "\n".join([f"![Generated Image]({u})" for u in images])
                await __event_emitter__(
                    {"type": "message", "data": {"content": f"\n\n{md}\n"}}
                )
                await __event_emitter__(
                    {"type": "status", "data": {"description": "Image generated!", "done": True}}
                )
            else:
                await __event_emitter__(
                    {"type": "status", "data": {"description": "No image returned.", "done": True}}
                )
        except Exception as e:
            await __event_emitter__(
                {"type": "status", "data": {"description": f"Error: {str(e)[:80]}", "done": True}}
            )
