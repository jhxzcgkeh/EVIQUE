def test_import_evique():
    import evique
    assert evique.MODEL_NAME == "EVIQUE"

def test_public_wrappers_import():
    from evique.evidence import EvidencePacker, QueryPlanner, load_evidence_graph
    from evique.retrieval import EvidenceRetriever
    from evique.views import load_scope_view
    assert EvidencePacker and QueryPlanner and EvidenceRetriever and load_scope_view and load_evidence_graph
