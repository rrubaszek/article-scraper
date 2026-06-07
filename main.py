import os
import json
import re
import logging
import time
import pandas as pd
import numpy as np

from typing import Dict, Set, final
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

from src.scraper import Scraper
from src.generator import ArticleGenerator
from src.batch_generator import AIBatchGenerator
from sklearn.feature_extraction.text import TfidfVectorizer


MAX_URLS = 5000
DATASETS = {
    "polskieradio": {
        "sections": [
            "https://polskieradio.pl/10/Polska",
            "https://polskieradio.pl/5/3",
            "https://polskieradio.pl/5/1",
            "https://polskieradio.pl/7"
        ],
        "blocklist": [
            "/395",
            "/399",
            "/400"
        ],
        "headers": {"User-Agent": "Mozilla/5.0 (dataset research bot)"},
        "regex": re.compile(r"polskieradio\.pl/.+/\d+"),
    },
    "onet": {
        "sections": [
            "https://www.onet.pl/",
            "https://wiadomosci.onet.pl/",
            "https://www.onet.pl/sport"
        ],
        "blocklist": None,
        "headers": None,
        "regex": re.compile(r"onet\.pl|wiadomosci\.onet\.pl"),
    },
}

logger = logging.getLogger(__name__)

def get_dataset_config(dataset_type: str) -> Dict:
    if dataset_type not in DATASETS:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    return DATASETS[dataset_type]


def collect_urls(scraper: Scraper, dataset_type: str) -> Set[str]:
    config = get_dataset_config(dataset_type)
    all_urls = set()

    if dataset_type == "polskieradio":
        # Use crawl method for polskieradio
        try:
            urls = scraper.crawl(
                config["sections"],
                regex=config["regex"],
                headers=config["headers"],
                blocklist=config["blocklist"],
                max_pages=MAX_URLS
            )
            all_urls.update(urls)
        except Exception as e:
            logger.warning(f"Failed to crawl {dataset_type}: {e}")
    else:
        # Use get_links method for other datasets
        for section in config["sections"]:
            try:
                urls = scraper.get_links(section, headers=config["headers"], regex=config["regex"], blocklist=config["blocklist"])
                all_urls.update(urls)
            except Exception as e:
                logger.warning(f"Failed to collect URLs from {section}: {e}")

    logger.info(f"Collected {len(all_urls)} URLs for {dataset_type}")
    return all_urls


def extract_dataset(scraper: Scraper, dataset_type: str, urls: Set[str]) -> list:
    dataset = []

    for url in tqdm(list(urls)[:MAX_URLS], desc=f"Extracting {dataset_type}"):
        try:
            item = scraper.extract(url, dataset_type)
            if item:
                dataset.append(item)
        except Exception as e:
            logger.debug(f"Failed to extract {url}: {e}")
            continue

    return dataset


def save_dataset(dataset: list, dataset_type: str) -> None:
    filename = f"{dataset_type}_dataset.jsonl"
    with open(f"dataset/{filename}", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(dataset)} items to {filename}")


def build_dataset(dataset_type: str, scraper: Scraper) -> None:
    urls = collect_urls(scraper, dataset_type)
    dataset = extract_dataset(scraper, dataset_type, urls)
    save_dataset(dataset, dataset_type)


def extract_tfidf_topics(texts, top_k=1):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=None  # Polish stopwords optional (see below)
    )

    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    topics = []

    for i in range(X.shape[0]):
        row = X[i].toarray().flatten()
        top_indices = row.argsort()[-top_k:][::-1]
        topics.append(" ".join(terms[top_indices]))

    return topics


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main(force: bool = False):
    scraper = Scraper()

    for dataset_type in DATASETS:
        output_file = f"{dataset_type}_dataset.jsonl"
        if not os.path.exists(f"dataset/{output_file}") or force==True:
            build_dataset(dataset_type, scraper)
        else:
            logger.info(f"Skipping {dataset_type}, {output_file} already exists")
    
    
    human_dfs = []

    for dataset_type in DATASETS:
        path = f"dataset/{dataset_type}_dataset.jsonl"
        df = load_jsonl(path)
        df["source_type"] = "human"
        human_dfs.append(df)

    human = pd.concat(human_dfs, ignore_index=True)
    human = human.dropna(subset=["text"])
    human = human[human["text"].str.len() > 400]
    
    load_dotenv()
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    MODEL = "gemini-2.5-flash"  # fast + cheap + good enough for datasets
    generator = ArticleGenerator(client, MODEL)
    
    topics = extract_tfidf_topics(human["text"].tolist(), top_k=2)
    
    ai_runner = AIBatchGenerator(
        generator=generator,
        batch_size=10,
        sleep_time=60,
        output_path="dataset/final_dataset.jsonl"
    )

    ai_rows = ai_runner.generate(topics)
    ai_df = pd.DataFrame(ai_rows)
    ai_df["source_type"] = "ai"
    
    min_size = min(len(human), len(ai_df))

    final = pd.concat([
        human.sample(min_size, random_state=42),
        ai_df.sample(min_size, random_state=42)
    ])

    final = final.sample(frac=1, random_state=42)
    
    os.makedirs("dataset", exist_ok=True)

    final.to_json(
        "dataset/final_dataset.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    logger.info(f"Final dataset size: {len(final)}")
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(force=False)
