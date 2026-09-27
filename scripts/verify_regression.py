"""Manual end-to-end text chat check against a running backend."""
import json
import sys

import httpx


API_URL = "http://127.0.0.1:8000"


def main() -> int:
    with httpx.Client(timeout=30.0) as client:
        conversation = client.post(
            f"{API_URL}/conversations",
            json={"title": "Regression check"},
        )
        conversation.raise_for_status()
        conversation_id = conversation.json()["data"]["id"]

        tokens: list[str] = []
        with client.stream(
            "POST",
            f"{API_URL}/chat/",
            json={
                "query": "Reply with exactly: chat is working",
                "model": "gen-labs-scout",
                "conversation_id": conversation_id,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    tokens.append(event.get("token", ""))

    output = "".join(tokens).strip()
    print(output)
    return 0 if output else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Regression verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
