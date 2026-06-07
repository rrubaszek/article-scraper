import os
import json
import re
import logging
from typing import Dict, Optional, Set
from tqdm import tqdm

from src.scraper import Scraper

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


def main(force: bool = False):
    logging.basicConfig(level=logging.INFO)
    scraper = Scraper()

    for dataset_type in DATASETS:
        output_file = f"{dataset_type}_dataset.jsonl"
        if not os.path.exists(output_file) or force==True:
            build_dataset(dataset_type, scraper)
        else:
            logger.info(f"Skipping {dataset_type}, {output_file} already exists")


if __name__ == "__main__":
    main(force=True)
