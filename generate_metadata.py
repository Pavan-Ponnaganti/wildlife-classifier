"""
iucn_cache.py — Optionally refresh iucn_cache.json using the Gemini API.
Run: python iucn_cache.py --api-key YOUR_KEY
     or set GEMINI_API_KEY env variable.
Skips animals already in the cache.
"""

import argparse
import json
import os
import time
from pathlib import Path

ANIMALS = [
    "antelope","badger","bat","bear","bee","beetle","bison","boar","butterfly",
    "cat","caterpillar","chimpanzee","cockroach","cow","coyote","crab","crow",
    "deer","dog","dolphin","donkey","dragonfly","duck","eagle","elephant",
    "flamingo","fly","fox","goat","goldfish","goose","gorilla","grasshopper",
    "hamster","hare","hedgehog","hippopotamus","hornbill","horse","hummingbird",
    "hyena","jellyfish","kangaroo","koala","ladybugs","leopard","lion","lizard",
    "lobster","mosquito","moth","mouse","octopus","okapi","orangutan","otter",
    "owl","ox","oyster","panda","parrot","pelecaniformes","penguin","pig",
    "pigeon","porcupine","possum","raccoon","rat","reindeer","rhinoceros",
    "sandpiper","seahorse","seal","shark","sheep","snake","sparrow","squid",
    "squirrel","starfish","swan","tiger","turkey","turtle","whale","wolf",
    "wombat","woodpecker","zebra",
]

CACHE_FILE = Path(__file__).parent / "iucn_cache.json"

PROMPT_TEMPLATE = """
For the animal '{animal}', provide:
1. IUCN Red List conservation status (use one of: EX, EW, CR, EN, VU, NT, LC, DD)
2. One fascinating fun fact (1-2 sentences, engaging for a nature app)

Respond ONLY with valid JSON in this exact format:
{{"iucn_status": "LC", "fun_fact": "Your fun fact here."}}
"""


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_for_animal(client, animal: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(animal=animal)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    text = response.text.strip()
    # strip markdown code blocks if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def main():
    parser = argparse.ArgumentParser(description="Refresh IUCN cache via Gemini")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Gemini API key")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already cached")
    args = parser.parse_args()

    if not args.api_key:
        print("No API key provided. Set GEMINI_API_KEY or use --api-key. Exiting.")
        return

    from google import genai
    client = genai.Client(api_key=args.api_key)

    cache = load_cache()
    updated = 0

    for animal in ANIMALS:
        if not args.force and animal in cache:
            print(f"  [skip] {animal} already cached")
            continue
        try:
            print(f"  [fetch] {animal} ...")
            data = fetch_for_animal(client, animal)
            cache[animal] = data
            save_cache(cache)
            updated += 1
            time.sleep(0.5)  # rate-limit courtesy
        except Exception as e:
            print(f"  [error] {animal}: {e}")

    print(f"\nDone. Updated {updated} entries. Cache saved to {CACHE_FILE}")


if __name__ == "__main__":
    main()
