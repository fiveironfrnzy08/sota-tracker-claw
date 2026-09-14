"""
Artificial Analysis scraper using Playwright.

Extracts rich model data from the RSC (React Server Components) payload
embedded in the leaderboard page. This captures intelligence scores,
speed metrics, pricing, and benchmark data that aren't available from
simple table scraping.

Source: https://artificialanalysis.ai/leaderboards/models
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeout

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.classification import is_open_source

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"


class ArtificialAnalysisScraper:
    """Scrape rich model data from Artificial Analysis RSC payload."""

    BASE_URL = "https://artificialanalysis.ai"
    LLM_URL = f"{BASE_URL}/leaderboards/models"
    TIMEOUT = 45000

    # Fields to extract from each model in the RSC payload
    EXTRACT_FIELDS = [
        "id", "slug", "name", "short_name", "model_family_slug",
        "intelligence_index", "coding_index", "agentic_index",
        "is_open_weights", "reasoning_model", "frontier_model",
        "release_date", "context_window_tokens", "output_tokens",
        "price_1m_input_tokens", "price_1m_output_tokens", "price_1m_blended_3_to_1",
        "size_class", "deprecated", "deleted",
        # Benchmark scores
        "gpqa", "hle", "humaneval", "scicode", "aime", "aime25",
        "math_500", "mmlu_pro", "livecodebench", "ifbench",
        "omniscience", "gdpval", "tau2", "terminalbench_hard", "lcr",
    ]

    TIMESCALE_FIELDS = [
        "median_output_speed",
        "median_time_to_first_chunk",
        "median_estimated_total_seconds_for_100_output_tokens",
        "percentile_05_output_speed",
        "percentile_95_output_speed",
    ]

    def __init__(self, headless: bool = True):
        self.headless = headless

    def scrape(self) -> dict:
        """
        Scrape the LLM leaderboard, extracting full model data from RSC payload.

        Returns dict with source metadata and list of models with rich metrics.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            try:
                page = browser.new_page()
                page.set_default_timeout(self.TIMEOUT)

                print(f"Navigating to {self.LLM_URL}...")
                page.goto(self.LLM_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(5000)  # Let RSC chunks load

                models = self._extract_rsc_models(page)

                return {
                    "source": "artificial_analysis",
                    "category": "llm",
                    "url": self.LLM_URL,
                    "scraped_at": datetime.now().isoformat(),
                    "model_count": len(models),
                    "models": models,
                }

            except PlaywrightTimeout as e:
                print(f"Timeout: {e}")
                return {"source": "artificial_analysis", "error": str(e), "models": []}
            except Exception as e:
                print(f"Error: {e}")
                return {"source": "artificial_analysis", "error": str(e), "models": []}
            finally:
                browser.close()

    def _extract_rsc_models(self, page: Page) -> list[dict]:
        """
        Extract model data from Next.js RSC flight payload.

        The page embeds ~5MB of JSON in self.__next_f.push() script tags.
        We find the chunk containing model data, unescape the RSC encoding,
        and parse individual model objects.
        """
        # Decode the Flight envelope with JSON rather than manually unescaping it.
        # Model arrays can span script chunks, and selector metadata can precede
        # the separate rich leaderboard array.
        scripts = page.locator("script").all_text_contents()
        raw_models = self._parse_rsc_models(scripts)

        print(f"  Extracted {len(raw_models)} raw models from RSC payload")

        # Transform to our format
        models = []
        for raw in raw_models:
            if not isinstance(raw, dict):
                continue
            if raw.get("deleted") or raw.get("deprecated"):
                continue
            name = raw.get("name")
            if not name:
                continue

            model = self._transform_model(raw)
            if model:
                models.append(model)

        print(f"  Transformed {len(models)} active models")
        return models

    @staticmethod
    def _parse_rsc_models(scripts: list[str]) -> list[dict]:
        chunks = []
        decoder = json.JSONDecoder(object_hook=lambda obj: {
            key: None if value == "$undefined" else value
            for key, value in obj.items()
        })
        for script in scripts:
            match = re.search(r"self\.__next_f\.push\(\s*", script)
            if not match:
                continue
            try:
                envelope, _ = decoder.raw_decode(script[match.end():])
            except ValueError:
                continue
            if (isinstance(envelope, list) and len(envelope) > 1
                    and envelope[0] == 1 and isinstance(envelope[1], str)):
                chunks.append(envelope[1])

        payload = "".join(chunks)
        metadata = {}
        rich_models = {}
        for match in re.finditer(r'"models"\s*:\s*(?=\[)', payload):
            try:
                rows, _ = decoder.raw_decode(payload[match.end():])
            except ValueError:
                continue
            for row in rows:
                if not isinstance(row, dict) or not row.get("name"):
                    continue
                key = row.get("slug") or row["name"]
                if "intelligenceIndex" in row or "intelligence_index" in row:
                    rich_models[key] = row
                else:
                    metadata[key] = row
        return [{**metadata.get(key, {}), **row}
                for key, row in rich_models.items()]

    def _transform_model(self, raw: dict) -> Optional[dict]:
        """Transform a raw RSC model object into our storage format."""
        name = raw.get("name", "")
        ts = raw.get("timescaleData") or {}

        def pick(*keys):
            for key in keys:
                val = raw.get(key)
                if val is not None:
                    return val
            return None

        def pick_ts(*keys):
            for key in keys:
                val = ts.get(key)
                if val is not None:
                    return val
            return pick(*keys)

        # Build rich metrics dict
        metrics = {
            "source": "artificial_analysis",
            "scraped_at": datetime.now().isoformat(),
        }

        # Intelligence & benchmark scores
        field_map = {
            "intelligence_index": ("intelligence_index", "intelligenceIndex"),
            "coding_index": ("coding_index", "codingIndex"),
            "agentic_index": ("agentic_index", "agenticIndex"),
            "gpqa": ("gpqa",),
            "hle": ("hle",),
            "humaneval": ("humaneval", "humanEval"),
            "scicode": ("scicode", "sciCode"),
            "aime": ("aime",),
            "aime25": ("aime25",),
            "math_500": ("math_500", "math500"),
            "mmlu_pro": ("mmlu_pro", "mmluPro"),
            "livecodebench": ("livecodebench", "liveCodeBench"),
            "ifbench": ("ifbench", "ifBench"),
            "omniscience": ("omniscience",),
            "gdpval": ("gdpval", "gdpvalNormalized"),
            "tau2": ("tau2",),
            "terminalbench_hard": ("terminalbench_hard", "terminalbenchHard"),
            "lcr": ("lcr",),
        }
        for metric_name, keys in field_map.items():
            val = pick(*keys)
            if val is not None:
                metrics[metric_name] = val

        # Speed / performance (from timescaleData)
        speed_map = {
            "median_output_speed": ("median_output_speed", "medianOutputTokensPerSecond"),
            "median_time_to_first_chunk": ("median_time_to_first_chunk", "medianTimeToFirstTokenSeconds"),
            "median_estimated_total_seconds_for_100_output_tokens": (
                "median_estimated_total_seconds_for_100_output_tokens",
                "medianEndToEndResponseTimeSeconds",
            ),
            "percentile_05_output_speed": ("percentile_05_output_speed", "percentile05OutputTokensPerSecond"),
            "percentile_95_output_speed": ("percentile_95_output_speed", "percentile95OutputTokensPerSecond"),
        }
        for metric_name, keys in speed_map.items():
            val = pick_ts(*keys)
            if val is not None:
                metrics[metric_name] = val

        # Model metadata
        metadata_map = {
            "model_family_slug": ("model_family_slug", "modelFamilySlug"),
            "size_class": ("size_class", "sizeClass"),
            "reasoning_model": ("reasoning_model", "reasoningModel", "isReasoning"),
            "frontier_model": ("frontier_model", "frontierModel"),
            "context_window_tokens": ("context_window_tokens", "contextWindowTokens"),
            "output_tokens": ("output_tokens", "outputTokens"),
            "short_name": ("short_name", "shortName"),
            "slug": ("slug",),
        }
        for metric_name, keys in metadata_map.items():
            val = pick(*keys)
            if val is not None:
                metrics[metric_name] = val

        return {
            "name": name,
            "category": "llm_api",
            "is_open_source": pick("is_open_weights", "isOpenWeights") or False,
            "release_date": pick("release_date", "releaseDate"),
            "intelligence_index": pick("intelligence_index", "intelligenceIndex"),
            "median_output_speed": pick_ts("median_output_speed", "medianOutputTokensPerSecond"),
            "median_ttft": pick_ts("median_time_to_first_chunk", "medianTimeToFirstTokenSeconds"),
            "price_1m_input": pick("price_1m_input_tokens", "price1mInputTokens"),
            "price_1m_output": pick("price_1m_output_tokens", "price1mOutputTokens"),
            "context_window": pick("context_window_tokens", "contextWindowTokens"),
            "model_family_slug": pick("model_family_slug", "modelFamilySlug"),
            "reasoning_model": pick("reasoning_model", "reasoningModel", "isReasoning") or False,
            "metrics": metrics,
        }


def scrape_artificial_analysis() -> dict:
    """Convenience function."""
    scraper = ArtificialAnalysisScraper()
    return scraper.scrape()


if __name__ == "__main__":
    print("Scraping Artificial Analysis (full RSC extraction)...")
    result = scrape_artificial_analysis()

    if result.get("error"):
        print(f"Error: {result['error']}")
    else:
        print(f"\nScraped {result['model_count']} models at {result['scraped_at']}")

        # Show top 10 by intelligence
        models = sorted(
            [m for m in result["models"] if m.get("intelligence_index")],
            key=lambda m: m["intelligence_index"],
            reverse=True,
        )
        print("\nTop 10 by Intelligence Index:")
        for i, m in enumerate(models[:10], 1):
            speed = m.get("median_output_speed")
            speed_str = f"{speed:.0f} t/s" if speed else "N/A"
            price = m.get("price_1m_output")
            price_str = f"${price:.2f}/M" if price else "N/A"
            print(f"  {i}. {m['name']}: IQ={m['intelligence_index']:.1f}  Speed={speed_str}  Price={price_str}")

        # Show top 10 by speed
        fast = sorted(
            [m for m in result["models"] if m.get("median_output_speed")],
            key=lambda m: m["median_output_speed"],
            reverse=True,
        )
        print("\nTop 10 by Output Speed:")
        for i, m in enumerate(fast[:10], 1):
            iq = m.get("intelligence_index")
            iq_str = f"IQ={iq:.1f}" if iq else "IQ=N/A"
            print(f"  {i}. {m['name']}: {m['median_output_speed']:.0f} t/s  {iq_str}")

        # Save
        output_path = DATA_DIR / "aa_llm_latest.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {output_path}")
