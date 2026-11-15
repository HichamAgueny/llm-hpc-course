from datasets import load_dataset

print("--> Downloading English C4 dataset slice on the login node...")
# Pulling the exact file from your original prompt
dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
)

# Select the top 1024 text examples
calibration_dataset = dataset.select(range(1024))["text"]

print("--> Writing to calib_english.txt...")
with open("calib_english.txt", "w", encoding="utf-8") as f:
    for text_block in calibration_dataset:
        # Strip internal newlines to keep it one clean sample per line
        cleaned_text = text_block.replace("\n", " ").strip()
        if cleaned_text:
            f.write(cleaned_text + "\n")

print("--> Successfully generated calib_english.txt locally!")
