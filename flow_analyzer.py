#!/usr/bin/env python3
"""
generate_arcade_report.py

Reads flow.json and generates:
 - Markdown report with user actions + summary
 - Social image representing the flow

Uses OpenAI LLMs for actions + summary, and DALL·E for image generation.
Includes simple JSON-based caching.
"""

import json
import os
import hashlib
import base64
import io
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import requests
from PIL import Image

load_dotenv()  # Loads variables from .env into os.environ

OUT_DIR = Path("./arcade_output")
CACHE_DIR = Path("./.cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------- Load flow ----------
def load_flow():
    for p in [Path("/mnt/data/flow.json"), Path("flow.json")]:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("flow.json not found")


# ---------- Caching helpers ----------
def _cache_key(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def cache_load(key: str):
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


def cache_save(key: str, obj):
    f = CACHE_DIR / f"{key}.json"
    f.write_text(json.dumps(obj, indent=2))


# ---------- LLM actions + summary ----------
def analyze_with_llm(flow):
    prompt = f"""
You are analyzing an Arcade flow recording.
Here is the raw JSON (truncated if too long):

{json.dumps(flow, indent=2)[:8000]}

Tasks:
1. Extract a bullet list of user actions in human-readable form 
   (e.g. "Clicked on checkout", "Searched for 'scooter'").
2. Write a summary of what the user was trying to accomplish through the actions performed.

Return your response STRICTLY as JSON in this format:
{{
  "actions": ["...", "..."],
  "summary": "..."
}}
    """

    key = _cache_key(prompt + "ANALYSIS")
    cached = cache_load(key)
    if cached:
        return cached.get("actions", []), cached.get("summary", "")

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.3,
    )
    text = resp.output_text.strip()

    # Try to parse as JSON
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            parsed = json.loads(text[start:end])
        except Exception:
            parsed = {"actions": ["[Error parsing actions]"], "summary": text}

    cache_save(key, parsed)
    return parsed.get("actions", []), parsed.get("summary", "")


# ---------- DALL·E image ----------
def create_social_image(summary, out_path: Path):
    prompt = f"""
Create a professional, eye-catching landscape social media graphic that represents this flow and would help
drive engagement:
"{summary}"
Make it visually engaging and suitable for sharing on LinkedIn/Twitter.
    """

    key = _cache_key(prompt + "IMAGE")
    cached = cache_load(key)
    if cached and out_path.exists():
        return out_path

    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
    )

    # Either b64_json or URL
    if hasattr(resp.data[0], "b64_json") and resp.data[0].b64_json:
        img_bytes = base64.b64decode(resp.data[0].b64_json)
        img_data = Image.open(io.BytesIO(img_bytes))
    elif hasattr(resp.data[0], "url") and resp.data[0].url:
        url = resp.data[0].url
        r = requests.get(url)
        r.raise_for_status()
        img_data = Image.open(io.BytesIO(r.content))
    else:
        raise ValueError("No image returned by OpenAI API")

    # Resize to 1200x630 for social media cards
    img_data = img_data.resize((1200, 630), Image.LANCZOS)
    img_data.save(out_path)

    cache_save(key, {"cached": True})
    return out_path


# ---------- Markdown ----------
def generate_markdown(report_path: Path, flow, actions, summary, image_filename):
    lines = []
    lines.append(f"# Arcade Flow Analysis — {flow.get('name', 'Arcade Flow')}\n")
    lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    lines.append("## 1) User interactions (LLM)\n")
    for a in actions:
        lines.append(f"- {a}")
    lines.append("\n## 2) Human-friendly summary (LLM)\n")
    lines.append(summary)
    lines.append("\n## 3) Generated social image\n")
    lines.append(f"![social image]({image_filename})")
    lines.append("\n---\n")
    lines.append("### Raw metadata (truncated)\n")
    raw_json = json.dumps(flow, indent=2)
    lines.append("```json\n")
    lines.append(raw_json[:10000])
    lines.append("\n```\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------- Main ----------
def main():
    flow = load_flow()

    try:
        actions, summary = analyze_with_llm(flow)
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        actions, summary = [], "[Analysis failed]"

    image_path = OUT_DIR / "social_image.png"

    if summary:
        try:
            create_social_image(summary, image_path)
        except Exception as e:
            print(f"Image generation failed: {e}")

    report_path = OUT_DIR / "report.md"
    generate_markdown(report_path, flow, actions, summary, image_path.name)

    print("Created:")
    print(f"- {report_path.resolve()}")
    print(f"- {image_path.resolve()}")


if __name__ == "__main__":
    main()
