from datasets import load_dataset

dataset = load_dataset("wmcnicho/StudyChat")

dataset["train"].to_json("StudyChat.jsonl")

print("DONE — saved to StudyChat.jsonl")