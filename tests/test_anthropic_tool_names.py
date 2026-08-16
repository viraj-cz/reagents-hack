from reagents.llm.anthropic_client import _anthropic_tool_name


def test_canonical_names_are_mapped_to_provider_safe_aliases():
    assert _anthropic_tool_name("paperclip.search-papers") == (
        "paperclip__search-papers"
    )
