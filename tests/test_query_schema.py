from evique.utils.query_io import QueryDataset, QueryRecord, validate_query_dataset


def test_query_dataset_schema_validation():
    dataset = QueryDataset(
        schema_version="1.0",
        dataset="Warsaw",
        query_count=1,
        queries=[QueryRecord("warsaw_001", "Warsaw", "warsaw", "Find a bus.", {})],
    )
    assert validate_query_dataset(dataset, expected_count=1) == []
    assert dataset.to_dict()["queries"][0]["query"] == "Find a bus."
