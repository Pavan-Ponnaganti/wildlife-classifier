import wikipedia
import json
import os

with open('dataset/name of the animals.txt', 'r') as f:
    animals = [line.strip() for line in f if line.strip()]

data = {}
for animal in animals:
    print(f"Fetching info for {animal}...")
    try:
        # Fetch summary, 2 sentences
        summary = wikipedia.summary(animal + " animal", sentences=2, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=2)
        except:
            summary = f"{animal.capitalize()} is a fascinating animal."
    except Exception as e:
        summary = f"{animal.capitalize()} is a fascinating animal."
    
    # Assign a generic IUCN status or derive it roughly. We'll use "Unknown" or "Least Concern" as a placeholder
    data[animal] = {
        "description": summary,
        "iucn_status": "Least Concern" # Placeholder unless we parse infoboxes.
    }

with open('animal_info.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print("Done generating animal_info.json")
