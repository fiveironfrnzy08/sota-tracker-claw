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
        raw_models = page.evaluate("""
            (() => {
                const scripts = document.querySelectorAll('script');
                for (const s of scripts) {
                    const t = s.textContent;
                    if (!t || !t.includes('agentic_index') || t.length < 100000) continue;

                    // Unescape RSC double-encoding: \\" → "
                    let chunk = t;
                    chunk = chunk.replace(/\\\\\\\\"/g, '§EQ§');
                    chunk = chunk.replace(/\\\\"/g, '"');
                    chunk = chunk.replace(/§EQ§/g, '\\\\"');

                    // Extract individual model objects by finding { ... "intelligence_index": ... }
                    // Walk through and parse each top-level object in the models array
                    const models = [];
                    const marker = '"models":[{';
                    const mIdx = chunk.indexOf(marker);
                    if (mIdx === -1) continue;

                    let pos = mIdx + marker.length - 1; // start at the first {
                    while (pos < chunk.length && models.length < 500) {
                        // Skip whitespace and commas
                        while (pos < chunk.length && (chunk[pos] === ',' || chunk[pos] === ' ' || chunk[pos] === '\\n')) pos++;

                        if (chunk[pos] === ']') break; // end of array
                        if (chunk[pos] !== '{') break; // unexpected

                        // Find matching closing brace, respecting nesting and strings
                        let depth = 0;
                        let objStart = pos;
                        let inString = false;
                        for (let i = pos; i < chunk.length; i++) {
                            const ch = chunk[i];
                            if (inString) {
                                if (ch === '\\\\') { i++; continue; } // skip escaped char
                                if (ch === '"') inString = false;
                                continue;
                            }
                            if (ch === '"') { inString = true; continue; }
                            if (ch === '{') depth++;
                            if (ch === '}') {
                                depth--;
                                if (depth === 0) {
                                    const objStr = chunk.substring(objStart, i + 1);
                                    // Replace $undefined with null for valid JSON
                                    const cleaned = objStr.replace(/\\$undefined/g, 'null');
                                    try {
                                        const obj = JSON.parse(cleaned);
                                        if (obj.intelligence_index !== undefined || obj.name) {
                                            models.push(obj);
                                        }
                                    } catch(e) {
                                        // Skip unparseable objects
                                    }
                                    pos = i + 1;
                                    break;
                                }
                            }
                        }
                        if (depth !== 0) break; // malformed, bail
                    }
                    return models;
                }
                return [];
            })()
        """)

        if isinstance(raw_models, dict) and "parseError" in raw_models:
            print(f"  Parse error: {raw_models['parseError']}")
            return []

        if not isinstance(raw_models, list):
            print(f"  Unexpected result type: {type(raw_models)}")
            return []

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

    def _transform_model(self, raw: dict) -> Optional[dict]:
        """Transform a raw RSC model object into our storage format."""
        name = raw.get("name", "")
        ts = raw.get("timescaleData") or {}

        # Build rich metrics dict
        metrics = {
            "source": "artificial_analysis",
            "scraped_at": datetime.now().isoformat(),
        }

        # Intelligence & benchmark scores
        for field in ["intelligence_index", "coding_index", "agentic_index",
                       "gpqa", "hle", "humaneval", "scicode", "aime", "aime25",
                       "math_500", "mmlu_pro", "livecodebench", "ifbench",
                       "omniscience", "gdpval", "tau2", "terminalbench_hard", "lcr"]:
            val = raw.get(field)
            if val is not None:
                metrics[field] = val

        # Speed / performance (from timescaleData)
        for field in self.TIMESCALE_FIELDS:
            val = ts.get(field)
            if val is not None:
                metrics[field] = val

        # Model metadata
        for field in ["model_family_slug", "size_class", "reasoning_model",
                       "frontier_model", "context_window_tokens", "output_tokens",
                       "short_name", "slug"]:
            val = raw.get(field)
            if val is not None:
                metrics[field] = val

        return {
            "name": name,
            "category": "llm_api",
            "is_open_source": raw.get("is_open_weights", False),
            "release_date": raw.get("release_date"),
            "intelligence_index": raw.get("intelligence_index"),
            "median_output_speed": ts.get("median_output_speed"),
            "median_ttft": ts.get("median_time_to_first_chunk"),
            "price_1m_input": raw.get("price_1m_input_tokens"),
            "price_1m_output": raw.get("price_1m_output_tokens"),
            "context_window": raw.get("context_window_tokens"),
            "model_family_slug": raw.get("model_family_slug"),
            "reasoning_model": raw.get("reasoning_model", False),
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
