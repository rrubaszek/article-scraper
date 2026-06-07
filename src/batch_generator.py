import time
import json
from tqdm import tqdm

class AIBatchGenerator:
    def __init__(self, generator, batch_size=10, sleep_time=30, output_path="dataset/final_dataset.jsonl"):
        self.generator = generator
        self.batch_size = batch_size
        self.sleep_time = sleep_time
        self.output_path = output_path

        self.generated = []
        self._load_existing()

    def _load_existing(self):
        self.done_topics = set()

        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if item.get("title"):
                        self.done_topics.add(item["title"])
        except FileNotFoundError:
            pass

    def _save_one(self, item):
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def generate(self, topics):
        start_index = len(self.generated)

        for i in tqdm(range(start_index, len(topics), self.batch_size)):
            batch = topics[i:i + self.batch_size]

            print(f"\nGenerating batch {i}-{i+len(batch)}")

            for topic in batch:
                if topic in self.done_topics:
                    continue
            
                try:
                    item = self.generator.generate_article(topic)
                    self.generated.append(item)
                    self._save_one(item)

                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(self.sleep_time)

            # IMPORTANT: cooldown after each batch
            print(f"Sleeping {self.sleep_time}s to avoid quota limits...")
            time.sleep(self.sleep_time)

        return self.generated