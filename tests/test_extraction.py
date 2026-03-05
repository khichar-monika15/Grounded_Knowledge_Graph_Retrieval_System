"""Tests for extraction pipeline. Use mocks — do NOT call real LLM API in tests."""
import pytest
import json


class TestPromptBuilder:
    def test_builds_prompt_with_all_fields(self, sample_email_raw):
        from pipeline.extraction import build_prompt
        prompt = build_prompt(sample_email_raw)
        assert "jeff.skilling@enron.com" in prompt
        assert "ken.lay@enron.com" in prompt
        assert "California situation" in prompt
        assert "JSON" in prompt

    def test_prompt_includes_schema_instructions(self, sample_email_raw):
        from pipeline.extraction import build_prompt
        prompt = build_prompt(sample_email_raw)
        assert "entities" in prompt
        assert "claims" in prompt
        assert "supporting_excerpt" in prompt


class TestResponseParsing:
    def test_parse_valid_response(self, sample_valid_llm_response):
        from pipeline.extraction import parse_llm_response
        result = parse_llm_response(sample_valid_llm_response)
        assert "entities" in result
        assert "claims" in result
        assert len(result["entities"]) > 0
        assert len(result["claims"]) > 0

    def test_parse_malformed_response_returns_none(self, sample_malformed_llm_response):
        from pipeline.extraction import parse_llm_response
        result = parse_llm_response(sample_malformed_llm_response)
        assert result is None

    def test_parse_response_with_markdown_fences(self, sample_extraction_response):
        """LLMs sometimes wrap JSON in ```json ... ``` — parser should handle this."""
        from pipeline.extraction import parse_llm_response
        wrapped = f"```json\n{json.dumps(sample_extraction_response)}\n```"
        result = parse_llm_response(wrapped)
        assert result is not None
        assert "entities" in result


class TestExtractionValidation:
    def test_validate_extraction_output(self, sample_extraction_response):
        from pipeline.extraction import validate_extraction
        errors = validate_extraction(sample_extraction_response)
        assert errors == []

    def test_validate_missing_entities_key(self):
        from pipeline.extraction import validate_extraction
        errors = validate_extraction({"claims": []})
        assert len(errors) > 0

    def test_validate_claim_without_excerpt(self, sample_extraction_response):
        from pipeline.extraction import validate_extraction
        bad = sample_extraction_response.copy()
        bad["claims"] = [c.copy() for c in bad["claims"]]
        bad["claims"][0]["supporting_excerpt"] = ""
        errors = validate_extraction(bad)
        assert len(errors) > 0

    def test_validate_entity_without_type(self, sample_extraction_response):
        from pipeline.extraction import validate_extraction
        bad = sample_extraction_response.copy()
        bad["entities"] = [e.copy() for e in bad["entities"]]
        bad["entities"][0] = {k: v for k, v in bad["entities"][0].items() if k != "type"}
        errors = validate_extraction(bad)
        assert len(errors) > 0


class TestQuotedContentStripping:
    def test_strip_quoted_lines(self):
        from pipeline.extraction import strip_quoted_content
        body = "Andy please see below.\n> Original message here\n> More quoted text"
        clean, quoted = strip_quoted_content(body)
        assert "Andy please see below" in clean
        assert ">" not in clean
        assert "Original message here" in quoted

    def test_no_quoted_content(self):
        from pipeline.extraction import strip_quoted_content
        body = "Just a normal email with no quotes."
        clean, quoted = strip_quoted_content(body)
        assert clean == body
        assert quoted == ""


class TestExtractEmail:
    def test_extract_email_with_mock_llm(self, sample_email_raw, sample_valid_llm_response, mocker):
        from pipeline.extraction import extract_email
        mock_call = mocker.patch("pipeline.extraction.call_llm", return_value=sample_valid_llm_response)
        result = extract_email(sample_email_raw)
        assert result is not None
        assert "entities" in result
        mock_call.assert_called_once()

    def test_extract_email_retries_on_failure(self, sample_email_raw, mocker):
        from pipeline.extraction import extract_email
        mock_call = mocker.patch("pipeline.extraction.call_llm",
            side_effect=["INVALID JSON", '{"entities": [], "claims": []}'])
        result = extract_email(sample_email_raw)
        assert mock_call.call_count == 2

    def test_extract_short_email_returns_minimal(self, sample_email_short, mocker):
        from pipeline.extraction import extract_email
        mock_call = mocker.patch("pipeline.extraction.call_llm",
            return_value='{"entities": [], "claims": []}')
        result = extract_email(sample_email_short)
        assert result is not None
