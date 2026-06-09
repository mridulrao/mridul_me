import re
import json
from datetime import datetime
from pathlib import Path

YOUR_NAME = "Mridul Rao"

WHATSAPP_PATTERN = re.compile(
    r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}:\d{2}\s?[APMapm]*?)\]\s([^:]+):\s(.*)$'
)

def parse_datetime(date_str, time_str):
    for fmt in ["%m/%d/%y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p"]:
        try:
            return datetime.strptime(f"{date_str} {time_str.upper()}", fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse datetime: {date_str} {time_str}")

def time_bucket(hour):
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "late_night"

def clean_message(msg):
    bad_phrases = [
        "<Media omitted>",
        "This message was deleted",
        "You deleted this message",
        "Missed voice call",
        "Missed video call",
    ]

    msg = msg.strip()

    for phrase in bad_phrases:
        if phrase in msg:
            return None

    return msg if msg else None

def parse_whatsapp_file(file_path):
    messages = []
    current = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            match = WHATSAPP_PATTERN.match(line)

            if match:
                date_str, time_str, sender, msg = match.groups()
                msg = clean_message(msg)

                if msg is None:
                    current = None
                    continue

                dt = parse_datetime(date_str, time_str)

                current = {
                    "datetime": dt,
                    "sender": sender.strip(),
                    "message": msg
                }

                messages.append(current)

            else:
                # multiline WhatsApp message
                if current:
                    extra = clean_message(line)
                    if extra:
                        current["message"] += "\n" + extra

    return messages

def merge_consecutive_messages(messages):
    merged = []

    for msg in messages:
        if (
            merged
            and merged[-1]["sender"] == msg["sender"]
            and merged[-1]["datetime"].date() == msg["datetime"].date()
        ):
            merged[-1]["message"] += "\n" + msg["message"]
            merged[-1]["end_datetime"] = msg["datetime"]
        else:
            merged.append({
                "sender": msg["sender"],
                "message": msg["message"],
                "datetime": msg["datetime"],
                "end_datetime": msg["datetime"]
            })

    return merged

def build_training_examples(merged_messages, your_name):
    examples = []

    for i in range(1, len(merged_messages)):
        current = merged_messages[i]
        previous = merged_messages[i - 1]

        # We only train on your replies
        if current["sender"] != your_name:
            continue

        # Previous block should be from friend
        if previous["sender"] == your_name:
            continue

        dt = current["datetime"]

        system_prompt = (
            f"You are {your_name}. "
            f"Day: {dt.strftime('%A')}. "
            f"Time category: {time_bucket(dt.hour)}. "
            f"Reply in casual Hinglish WhatsApp style. "
            f"You may reply with text, emojis, or multiple short messages separated by newlines."
        )

        example = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": previous["message"]
                },
                {
                    "role": "assistant",
                    "content": current["message"]
                }
            ],
            "metadata": {
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "day": dt.strftime("%A"),
                "time_bucket": time_bucket(dt.hour),
                "friend_name": previous["sender"]
            }
        }

        examples.append(example)

    return examples

def save_jsonl(examples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def save_json(examples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "_chat.txt"

    messages = parse_whatsapp_file(input_file)
    merged_messages = merge_consecutive_messages(messages)
    examples = build_training_examples(merged_messages, YOUR_NAME)

    save_jsonl(examples, "mridul_sft_dataset.jsonl")
    save_json(examples, "mridul_sft_dataset.json")

    print(f"Parsed messages: {len(messages)}")
    print(f"Merged message blocks: {len(merged_messages)}")
    print(f"Training examples created: {len(examples)}")