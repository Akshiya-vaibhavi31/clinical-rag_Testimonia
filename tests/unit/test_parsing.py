"""
Unit tests for src/clients/pubmed_client.py's XML parsing.

Covers the real DOI/PMCID extraction bug found earlier in this project:
the original code searched the entire article XML tree for ArticleIdList
elements, which matched not just the article's own IDs but also every
reference's IDs in its bibliography — silently returning the wrong DOI.

NOTE: this module requires `tenacity`, which may not be installed in
every environment. This test file will be skipped automatically if the
import fails, rather than causing the whole test run to error out.
"""

import pytest

pytest.importorskip("tenacity")

from src.clients.pubmed_client import parse_pubmed_xml


SAMPLE_XML_WITH_REFERENCES = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
<MedlineCitation>
<PMID>42572726</PMID>
<Article>
<ArticleTitle>Meta-Analysis of the Effect of Metformin</ArticleTitle>
<Abstract><AbstractText>Test abstract content.</AbstractText></Abstract>
<Journal><Title>Test Journal</Title><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>
</Article>
</MedlineCitation>
<PubmedData>
<ArticleIdList>
<ArticleId IdType="pubmed">42572726</ArticleId>
<ArticleId IdType="pmc">PMC13453383</ArticleId>
<ArticleId IdType="doi">10.2147/DMSO.S596986</ArticleId>
</ArticleIdList>
<ReferenceList>
<Reference>
<Citation>Some other cited paper</Citation>
<ArticleIdList>
<ArticleId IdType="doi">10.1016/S0140-6736(12)60283-9</ArticleId>
<ArticleId IdType="pmc">PMC3891203</ArticleId>
</ArticleIdList>
</Reference>
</ReferenceList>
</PubmedData>
</PubmedArticle>
</PubmedArticleSet>
"""


class TestParsePubmedXml:
    def test_extracts_basic_fields(self):
        articles = parse_pubmed_xml(SAMPLE_XML_WITH_REFERENCES)
        assert len(articles) == 1
        article = articles[0]
        assert article["pmid"] == "42572726"
        assert "Metformin" in article["title"]
        assert article["abstract"] == "Test abstract content."
        assert article["pub_year"] == "2026"

    def test_extracts_the_articles_own_doi_not_a_reference_doi(self):
        # Regression test for the real bug: the article's own DOI must be
        # returned, NOT one of the DOIs from its reference list.
        articles = parse_pubmed_xml(SAMPLE_XML_WITH_REFERENCES)
        assert articles[0]["doi"] == "10.2147/DMSO.S596986"
        assert articles[0]["doi"] != "10.1016/S0140-6736(12)60283-9"

    def test_extracts_the_articles_own_pmcid_not_a_reference_pmcid(self):
        articles = parse_pubmed_xml(SAMPLE_XML_WITH_REFERENCES)
        assert articles[0]["pmcid"] == "PMC13453383"
        assert articles[0]["pmcid"] != "PMC3891203"

    def test_missing_doi_and_pmcid_return_none_not_crash(self):
        xml_no_ids = SAMPLE_XML_WITH_REFERENCES.replace(
            '<ArticleId IdType="pmc">PMC13453383</ArticleId>\n<ArticleId IdType="doi">10.2147/DMSO.S596986</ArticleId>',
            "",
        )
        articles = parse_pubmed_xml(xml_no_ids)
        assert articles[0]["doi"] is None
        assert articles[0]["pmcid"] is None

    def test_empty_xml_returns_empty_list(self):
        empty_xml = "<?xml version='1.0'?><PubmedArticleSet></PubmedArticleSet>"
        assert parse_pubmed_xml(empty_xml) == []
