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
from typing import Iterable

# Add parent to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("SOTA_DATA_DIR", PROJECT_DIR / "data")).expanduser()
DB_PATH = Path(os.environ.get("SOTA_DB_PATH", DATA_DIR / "sota.db")).expanduser()
AA_EXPORT_PATH = Path(os.environ.get("SOTA_AA_EXPORT_PATH", DATA_DIR / "aa_llm_latest.json")).expanduser()
AA_MIN_MODELS = int(os.environ.get("SOTA_AA_MIN_MODELS", "100"))
AA_MIN_RICH_MODELS = int(os.environ.get("SOTA_AA_MIN_RICH_MODELS", "50"))

from utils.models import normalize_model_id
from utils.db import get_db_context
from scrapers.lmarena import LMArenaScraper
from scrapers.artificial_analysis import ArtificialAnalysisScraper


def unique(values: Iterable[str]) -> list[str]:
    """Return non-empty strings in first-seen order."""
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def effort_aliases(name: str) -> list[str]:
    """Map verbose AA effort names to the compact upstream names."""
    if "(" not in name:
        return []

    base = name.split("(", 1)[0].strip()
    lower = name.lower()
    aliases = []
    if "max effort" in lower or "adaptive reasoning" in lower:
        aliases.append(f"{base} (max)")
    if "high effort" in lower:
        aliases.append(f"{base} (high)")
    if "medium effort" in lower:
        aliases.append(f"{base} (medium)")
    if "low effort" in lower:
        aliases.append(f"{base} (low)")
    if "non-reasoning" in lower:
        aliases.append(base)
    return aliases


def model_id_candidates(model: dict) -> list[str]:
    """Generate possible DB IDs for the same model across source naming styles."""
    metrics = model.get("metrics") or {}
    names = [
        model.get("id"),
        model.get("name"),
        model.get("short_name"),
        model.get("slug"),
        metrics.get("short_name"),
        metrics.get("slug"),
    ]
    names.extend(effort_aliases(str(model.get("name") or "")))

    ids = []
    for value in names:
        if not value:
            continue
        text = str(value)
        ids.append(normalize_model_id(text))

        if text.endswith("-adaptive"):
            ids.append(f"{text[:-len('-adaptive')]}-(max)")

    return unique(ids)


def find_existing_model(db, candidates: list[str]):
    """Find the first existing model matching any candidate ID."""
    if not candidates:
        return None
    placeholders = ",".join("?" for _ in candidates)
    rows = db.execute(
        f"SELECT id, source FROM models WHERE id IN ({placeholders})",
        candidates,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}
    for candidate in candidates:
        if candidate in by_id:
            return by_id[candidate]
    return None


def rich_export_summary(data: dict) -> dict:
    """Summarize rich data coverage for status reporting."""
    models = data.get("models") or []
    rich_models = [
        model for model in models
        if model.get("intelligence_index") is not None
        or model.get("median_output_speed") is not None
        or model.get("price_1m_output") is not None
    ]
    return {
        "model_count": len(models),
        "rich_model_count": len(rich_models),
        "scraped_at": data.get("scraped_at"),
        "path": str(AA_EXPORT_PATH),
    }


def validate_rich_export(data: dict):
    """Reject obviously stale or partial rich exports before replacing cache."""
    summary = rich_export_summary(data)
    if summary["model_count"] < AA_MIN_MODELS:
        raise ValueError(f"AA rich export too small: {summary['model_count']} models")
    if summary["rich_model_count"] < AA_MIN_RICH_MODELS:
        raise ValueError(f"AA rich export has too few rich rows: {summary['rich_model_count']}")
    return summary


def write_json_atomic(path: Path, data: dict):
    """Write JSON using a temp file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


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

            candidates = model_id_candidates(model)
            model_id = candidates[0] if candidates else normalize_model_id(model["name"])

            # Check if model exists
            existing = find_existing_model(db, candidates)
            if existing:
                model_id = existing["id"]

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
                            price_per_1m_input = COALESCE(?, price_per_1m_input),
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
                        price_1m_input, price_per_1m_input, price_per_1m_output, context_window,
                        model_family_slug, reasoning_model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def update_cache_status(category: str, source: str, success: bool, error: str = None, fetched_at: str = None):
    """Update cache status table."""
    with get_db_context(DB_PATH) as db:
        db.execute(
            """
            INSERT OR REPLACE INTO cache_status (category, last_fetched, fetch_source, fetch_success, error_message)
            VALUES (?, ?, ?, ?, ?)
        """,
            (category, fetched_at or datetime.now().isoformat(), source, success, error),
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
        result = aa.scrape()
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
    aa_path = AA_EXPORT_PATH
    if not aa_path.exists():
        print("No aa_llm_latest.json found, skipping import")
        update_cache_status("llm_api_rich", "artificial_analysis_export", False, "No aa_llm_latest.json found")
        return

    print(f"Importing from {aa_path}...")
    with open(aa_path) as f:
        data = json.load(f)

    if data.get("models"):
        summary = rich_export_summary(data)
        result = {
            "models": data["models"],
            "model_count": len(data["models"]),
            "scraped_at": data.get("scraped_at", datetime.now().isoformat()),
        }
        count = update_models_from_scrape(result, "artificial_analysis")
        print(f"  Imported {count} models from AA export")
        update_cache_status(
            "llm_api_rich",
            data.get("source") or "artificial_analysis_export",
            summary["rich_model_count"] > 0,
            json.dumps(summary),
            fetched_at=summary["scraped_at"],
        )


def refresh_rich_export() -> dict:
    """Refresh the rich Artificial Analysis export and write it if valid."""
    print("Refreshing rich Artificial Analysis data...")
    scraper = ArtificialAnalysisScraper()
    data = scraper.scrape()

    if data.get("error"):
        raise RuntimeError(data["error"])

    summary = validate_rich_export(data)
    write_json_atomic(AA_EXPORT_PATH, data)
    print(
        f"  Saved {summary['model_count']} AA models "
        f"({summary['rich_model_count']} rich) to {AA_EXPORT_PATH}"
    )
    update_cache_status(
        "llm_api_rich",
        "artificial_analysis",
        True,
        json.dumps(summary),
        fetched_at=summary["scraped_at"],
    )
    return data


def refresh_rich_and_import():
    """Refresh rich AA data, then import it into the DB."""
    try:
        refresh_rich_export()
    except Exception as e:
        update_cache_status("llm_api_rich", "artificial_analysis", False, str(e))
        raise
    import_from_exports()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA data scrapers")
    parser.add_argument(
        "--export", action="store_true", help="Export to JSON/CSV after scraping"
    )
    parser.add_argument(
        "--import-only", action="store_true", help="Import from existing JSON exports (no scraping)"
    )
    parser.add_argument(
        "--refresh-rich", action="store_true", help="Refresh rich AA export and import it"
    )
    args = parser.parse_args()

    if args.refresh_rich:
        refresh_rich_and_import()
    elif args.import_only:
        import_from_exports()
    else:
        run_all_scrapers(export=args.export)
