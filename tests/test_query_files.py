import csv
from pathlib import Path

from evique.utils.query_io import load_query_file, summarize_dataset

EXPECTED = {
    "warsaw": ("Warsaw", 30),
    "bellevue": ("Bellevue", 25),
    "qvhighlights": ("QVHighlights", 20),
    "beach": ("Beach", 21),
}


def test_normalized_query_files_are_valid_json_and_schema_consistent():
    total = 0
    all_ids: set[str] = set()
    for slug, (dataset_name, expected_count) in EXPECTED.items():
        dataset = load_query_file(Path("data/queries") / f"{slug}.json")
        assert dataset.schema_version == "1.0"
        assert dataset.dataset == dataset_name
        assert dataset.query_count == len(dataset.queries) == expected_count
        total += dataset.query_count
        for record in dataset.queries:
            assert record.query_id
            assert record.query_id not in all_ids
            all_ids.add(record.query_id)
            assert record.dataset == dataset_name
            assert record.query
            assert isinstance(record.metadata, dict)
    assert total == 96


def test_qvhighlights_missing_video_ids_are_explicitly_reported():
    dataset = load_query_file("data/queries/qvhighlights.json")
    summary = summarize_dataset(dataset)
    assert dataset.query_count == 20
    assert summary["missing_video_ids"] == 20
    assert summary["video_count"] == 0
    report = Path("data/queries/QVHIGHLIGHTS_MAPPING_REPORT.md")
    assert report.exists()
    assert "No verified 20/20 query-to-video mapping was found" in report.read_text(encoding="utf-8")


def test_manifest_matches_query_files():
    rows = {row["dataset"]: row for row in csv.DictReader(Path("results/paper/query_manifest.csv").open("r", encoding="utf-8"))}
    assert rows["Warsaw"]["status"] == "ready"
    assert rows["Bellevue"]["status"] == "ready"
    assert rows["Beach"]["status"] == "ready"
    assert rows["QVHighlights"]["status"] == "missing_video_id"
    assert rows["QVHighlights"]["current_query_count"] == "20"
