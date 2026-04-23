#!/usr/bin/env python3
"""
Unified SOTA data scraper.

Runs all scrapers and updates the database with fresh data.
Designed to be run via GitHub Actions or cron.

Usage:
    python scrapers/run_all.py           # Run all scrapers
    python scrapers/run_all.py --export  # Also export to JSON/CSV
"""

import argparse
import json
import csv
import os
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("SOTA_DATA_DIR", PROJECT_DIR / "data")).expanduser()
DB_PATH = Path(os.environ.get("SOTA_DB_PATH", DATA_DIR / "sota.db")).expanduser()

from utils.models import normalize_model_id
from utils.db import get_db_context
from scrapers.lmarena import LMArenaScraper
from scrapers.artificial_analysis import ArtificialAnalysisScraper


def update_models_from_scrape(scraped_data: dict, source: str):
    """
    Update database with scraped data.

    Merges with existing data - doesn't overwrite manual entries.
    """
    models = scraped_data.get("models", [])
    if not models:
        print(f"  No models to update from {source}")
        return 0

    updated = 0
    inserted = 0

    with get_db_context(DB_PATH) as db:
        for model in models:
            # Validate required field
            if "name" not in model:
                print("  Warning: Skipping model without name field")
                continue

            model_id = normalize_model_id(model["name"])

            # Check if model exists
            existing = db.execute(
                "SELECT id, source FROM models WHERE id = ?", (model_id,)
            ).fetchone()

            # Build metrics JSON
            existing_metrics = {}
            if existing:
                existing_metrics_row = db.execute(
                    "SELECT metrics FROM models WHERE id = ?", (model_id,)
                ).fetchone()
                if existing_metrics_row and existing_metrics_row[0]:
                    try:
                        existing_metrics = json.loads(existing_metrics_row[0])
                    except json.JSONDecodeError:
                        pass

            # Merge new metrics into existing
            new_metrics = model.get("metrics", {})
            for key, val in new_metrics.items():
                if val is not None:
                    existing_metrics[key] = val
            if model.get("elo") is not None:
                existing_metrics["elo"] = model["elo"]
            existing_metrics["scraped_from"] = source
            existing_metrics["scraped_at"] = scraped_data.get("scraped_at")

            # Extract top-level columns from model dict (AA rich fields)
            intelligence_index = model.get("intelligence_index")
            median_output_speed = model.get("median_output_speed")
            median_ttft = model.get("median_ttft")
            price_1m_input = model.get("price_1m_input")
            price_1m_output = model.get("price_1m_output")
            context_window = model.get("context_window")
            model_family_slug = model.get("model_family_slug")
            reasoning_model = model.get("reasoning_model", False)
            release_date = model.get("release_date")

            if existing:
                # Allow artificial_analysis to enrich any existing model's metrics
                # (AA provides intelligence_index, speed, pricing that other sources lack)
                # Only skip updates if a manual entry would be overwritten by a lesser source
                can_update = (
                    existing["source"] in ["auto", source]
                    or source == "artificial_analysis"  # AA enriches all
                )
                if can_update:
                    # Preserve original source if this is an enrichment from AA
                    update_source = source if existing["source"] in ["auto", source] else existing["source"]
                    db.execute(
                        """
                        UPDATE models
                        SET sota_rank = COALESCE(?, sota_rank),
                            metrics = ?,
                            last_updated = ?,
                            source = ?,
                            is_open_source = ?,
                            release_date = COALESCE(?, release_date),
                            intelligence_index = COALESCE(?, intelligence_index),
                            median_output_speed = COALESCE(?, median_output_speed),
                            median_ttft = COALESCE(?, median_ttft),
                            price_1m_input = COALESCE(?, price_1m_input),
                            price_per_1m_output = COALESCE(?, price_per_1m_output),
                            context_window = COALESCE(?, context_window),
                            model_family_slug = COALESCE(?, model_family_slug),
                            reasoning_model = COALESCE(?, reasoning_model)
                        WHERE id = ?
                    """,
                        (
                            model.get("rank"),
                            json.dumps(existing_metrics),
                            datetime.now().isoformat(),
                            update_source,
                            model.get("is_open_source", True),
                            release_date,
                            intelligence_index,
                            median_output_speed,
                            median_ttft,
                            price_1m_input,
                            price_1m_output,
                            context_window,
                            model_family_slug,
                            reasoning_model,
                            model_id,
                        ),
                    )
                    updated += 1
            else:
                is_sota = 0 if source == "civitai" else 1
                db.execute(
                    """
                    INSERT INTO models (id, name, category, is_open_source, is_sota, sota_rank,
                        metrics, last_updated, source, release_date,
                        intelligence_index, median_output_speed, median_ttft,
                        price_1m_input, price_per_1m_output, context_window,
                        model_family_slug, reasoning_model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        model_id,
                        model["name"],
                        model.get("category", "llm_api"),
                        model.get("is_open_source", True),
                        is_sota,
                        model.get("rank"),
                        json.dumps(existing_metrics),
                        datetime.now().isoformat(),
                        source,
                        release_date,
                        intelligence_index,
                        median_output_speed,
                        median_ttft,
                        price_1m_input,
                        price_1m_output,
                        context_window,
                        model_family_slug,
                        reasoning_model,
                    ),
                )
                inserted += 1

        db.commit()

    print(f"  Updated {updated}, inserted {inserted} models from {source}")
    return updated + inserted


def update_cache_status(category: str, source: str, success: bool, error: str = None):
    """Update cache status table."""
    with get_db_context(DB_PATH) as db:
        db.execute(
            """
            INSERT OR REPLACE INTO cache_status (category, last_fetched, fetch_source, fetch_success, error_message)
            VALUES (?, ?, ?, ?, ?)
        """,
            (category, datetime.now().isoformat(), source, success, error),
        )
        db.commit()


def export_to_json():
    """Export all SOTA data to JSON."""
    with get_db_context(DB_PATH) as db:
        # Export all models
        rows = db.execute(
            """
            SELECT * FROM models WHERE is_sota = 1 ORDER BY category, sota_rank
        """
        ).fetchall()

        models = [dict(row) for row in rows]

    output = {
        "exported_at": datetime.now().isoformat(),
        "model_count": len(models),
        "models": models,
    }

    output_path = DATA_DIR / "sota_export.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported {len(models)} models to {output_path}")
    return output_path


def export_to_csv():
    """Export all SOTA data to CSV."""
    with get_db_context(DB_PATH) as db:
        rows = db.execute(
            """
            SELECT id, name, category, is_open_source, sota_rank, release_date, source, last_updated
            FROM models WHERE is_sota = 1 ORDER BY category, sota_rank
        """
        ).fetchall()

    output_path = DATA_DIR / "sota_export.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "name",
                "category",
                "is_open_source",
                "sota_rank",
                "release_date",
                "source",
                "last_updated",
            ]
        )
        for row in rows:
            writer.writerow(list(row))

    print(f"Exported {len(rows)} models to {output_path}")
    return output_path


def run_all_scrapers(export: bool = False):
    """Run all scrapers and update database."""
    print(f"=== SOTA Tracker Scraper Run: {datetime.now().isoformat()} ===\n")

    results = {}

    # 1. LMArena (Elo rankings for LLMs)
    print("1. Scraping LMArena...")
    try:
        lmarena = LMArenaScraper()
        result = lmarena.scrape()
        results["lmarena"] = result

        if result.get("models"):
            count = update_models_from_scrape(result, "lmarena")
            update_cache_status("llm_api", "lmarena", True)
            print(f"   SUCCESS: {result['model_count']} models scraped\n")
        else:
            update_cache_status("llm_api", "lmarena", False, result.get("error"))
            print(f"   FAILED: {result.get('error', 'No models')}\n")
    except Exception as e:
        print(f"   ERROR: {e}\n")
        update_cache_status("llm_api", "lmarena", False, str(e))

    # 2. Artificial Analysis LLM
    print("2. Scraping Artificial Analysis (LLM)...")
    try:
        aa = ArtificialAnalysisScraper()
        result = aa.scrape("llm")
        results["aa_llm"] = result

        if result.get("models"):
            count = update_models_from_scrape(result, "artificial_analysis")
            update_cache_status("llm_api", "artificial_analysis", True)
            print(f"   SUCCESS: {result['model_count']} models scraped\n")
        else:
            update_cache_status("llm_api", "artificial_analysis", False, result.get("error", "No models returned"))
            print(f"   FAILED: {result.get('error', 'No models')}\n")
    except Exception as e:
        print(f"   ERROR: {e}\n")
        update_cache_status("llm_api", "artificial_analysis", False, str(e))

    # 3. HuggingFace (Open LLM Leaderboard + Trending)
    print("3. Fetching HuggingFace data...")
    try:
        from fetchers.huggingface import HuggingFaceFetcher

        hf = HuggingFaceFetcher()

        # Open LLM Leaderboard
        result_llm = hf.fetch_llm_leaderboard()
        if result_llm:
            result = {
                "models": result_llm,
                "model_count": len(result_llm),
                "scraped_at": datetime.now().isoformat(),
            }
            update_models_from_scrape(result, "huggingface")
            update_cache_status("llm_local", "huggingface", True)
            results["hf_llm"] = result
            print(f"   SUCCESS: {len(result_llm)} local LLMs fetched")

        # Trending embeddings
        result_embed = hf.fetch_trending_models(task="feature-extraction", limit=10)
        if result_embed:
            for m in result_embed:
                m["category"] = "embeddings"
            result = {
                "models": result_embed,
                "model_count": len(result_embed),
                "scraped_at": datetime.now().isoformat(),
            }
            update_models_from_scrape(result, "huggingface")
            update_cache_status("embeddings", "huggingface", True)
            results["hf_embed"] = result
            print(f"   SUCCESS: {len(result_embed)} embeddings fetched\n")

    except Exception as e:
        print(f"   ERROR: {e}\n")
        update_cache_status("llm_local", "huggingface", False, str(e))
        update_cache_status("embeddings", "huggingface", False, str(e))

    # 4. Civitai (Image Generation)
    print("4. Fetching Civitai image models...")
    try:
        from scrapers.civitai import CivitaiScraper

        civitai = CivitaiScraper()
        result = civitai.scrape(model_type="Checkpoint", limit=20)
        results["civitai"] = result

        if result.get("models"):
            count = update_models_from_scrape(result, "civitai")
            update_cache_status("image_gen", "civitai", True)
            print(f"   SUCCESS: {result['model_count']} image models fetched\n")
        else:
            print(f"   FAILED: {result.get('error', 'No models')}\n")
    except Exception as e:
        print(f"   ERROR: {e}\n")
        update_cache_status("image_gen", "civitai", False, str(e))

    # Summary
    print("=== Summary ===")
    for source, result in results.items():
        status = "OK" if result.get("models") else "FAILED"
        count = result.get("model_count", 0)
        print(f"  {source}: {status} ({count} models)")

    # Export if requested
    if export:
        print("\n=== Exporting Data ===")
        export_to_json()
        export_to_csv()

    print(f"\nCompleted at {datetime.now().isoformat()}")

    return results


def import_from_exports():
    """Import models from existing JSON export files (no scraping).

    Used on startup to enrich the DB with AA data after git pull.
    """
    aa_path = Path(os.environ.get("SOTA_AA_EXPORT_PATH", DATA_DIR / "aa_llm_latest.json")).expanduser()
    if not aa_path.exists():
        print("No aa_llm_latest.json found, skipping import")
        return

    print(f"Importing from {aa_path}...")
    with open(aa_path) as f:
        data = json.load(f)

    if data.get("models"):
        result = {
            "models": data["models"],
            "model_count": len(data["models"]),
            "scraped_at": data.get("scraped_at", datetime.now().isoformat()),
        }
        count = update_models_from_scrape(result, "artificial_analysis")
        print(f"  Imported {count} models from AA export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA data scrapers")
    parser.add_argument(
        "--export", action="store_true", help="Export to JSON/CSV after scraping"
    )
    parser.add_argument(
        "--import-only", action="store_true", help="Import from existing JSON exports (no scraping)"
    )
    args = parser.parse_args()

    if args.import_only:
        import_from_exports()
    else:
        run_all_scrapers(export=args.export)
