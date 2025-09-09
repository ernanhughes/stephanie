# tools/split_conversations.py
import json
import os
import argparse

def split_conversation_array(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    for i, convo in enumerate(conversations, start=1):
        title = convo.get("title", f"conversation_{i}")
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:50]
        out_path = os.path.join(output_dir, f"{i:03d}_{safe_title}.json")

        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(convo, out_f, ensure_ascii=False, indent=2)

        print(f"✅ Wrote {out_path}")


def split_conversation_batches(input_path, output_dir, batch_size=10):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        out_path = os.path.join(output_dir, f"batch_{i//batch_size+1:03d}.json")

        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(batch, out_f, ensure_ascii=False, indent=2)

        print(f"✅ Wrote {out_path} with {len(batch)} conversations")


def main():
    parser = argparse.ArgumentParser(description="Split a JSON file of conversations into separate files.")
    parser.add_argument("--input", required=True, help="Path to input JSON file (e.g. data/chats/conversation.json)")
    parser.add_argument("--output", required=True, help="Output directory for split files")
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                        help="Split into single files (one per conversation) or batches")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size if mode=batch")

    args = parser.parse_args()

    if args.mode == "single":
        split_conversation_array(args.input, args.output)
    else:
        split_conversation_batches(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()
