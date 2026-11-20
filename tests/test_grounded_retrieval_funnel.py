from scripts.analyze_grounded_retrieval_funnel import primary_loss


def test_primary_loss_is_exclusive_and_uses_earliest_stage():
    row = {"corpus_exact": False, "shortlist_article": False, "scored_exact": False, "pack_exact": False}
    assert primary_loss(row) == "corpus_granularity"
    row["corpus_exact"] = True
    assert primary_loss(row) == "article_shortlist"
    row["shortlist_article"] = True
    assert primary_loss(row) == "within_article_candidate_pool"
    row["scored_exact"] = True
    assert primary_loss(row) == "evidence_pack"
    row["pack_exact"] = True
    assert primary_loss(row) == "available_to_generator"
