"""Attached tables: profiled, sealed from their own schema, and mounted.

The interesting failures are all at the edges of the profile rather than in the
upload itself, so most of this file is about what ends up in it:

* the projection manifest, which bills one obligation per leaf in
  `NativeProblem.inputs` and would be unsatisfiable if a profile nested lists,
* the derived seal, which replaced a text box and is therefore the only thing
  standing between a demigod and the user's native vocabulary,
* and the honest limit of that seal -- the raw file is mounted unsealed, and a
  test says so rather than leaving it to a docstring.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from reagents.god.anonymize import anonymize_problem
from reagents.god.transformer import projection_contract
from reagents.isolation import native_terms
from reagents.toy import toy_problem
from resolution.app import ResolutionApp
from resolution.attachments import (
    MAX_COLUMNS,
    VALUE_SEPARATOR,
    AttachmentError,
    AttachmentStore,
    profile_table,
    sealed_terms,
)
from resolution.runs import build_problem, with_attachments

SALES = b"""order_id,date,region,channel,gross_revenue,discount,refunded
ORD-1001,2025-04-03,west,partner,6000.0,300.0,false
ORD-1002,2025-04-07,east,direct,5200.0,260.0,false
ORD-1003,2025-07-11,north,partner,1925.0,96.25,true
ORD-1004,2025-08-15,west,direct,4300.0,215.0,false
ORD-1005,2025-09-02,east,partner,3100.0,155.0,true
"""


@pytest.fixture
def store(tmp_path):
    return AttachmentStore(tmp_path / "uploads")


def column(profile: dict, name: str) -> dict:
    return next(c for c in profile["columns"] if c["name"] == name)


# -- profiling -------------------------------------------------------------
def test_profile_describes_shape_without_carrying_rows(store) -> None:
    item = store.add("sales.csv", SALES)
    profile = item.profile

    assert profile["format"] == "csv"
    assert profile["rows"] == 5
    assert [c["name"] for c in profile["columns"]] == [
        "order_id",
        "date",
        "region",
        "channel",
        "gross_revenue",
        "discount",
        "refunded",
    ]

    # Numbers are summarised by range, never listed.
    revenue = column(profile, "gross_revenue")
    assert revenue["type"] == "number"
    assert (revenue["min"], revenue["max"]) == (1925, 6000)
    assert "values" not in revenue

    assert column(profile, "refunded")["type"] == "boolean"

    # No row survives anywhere in the profile.
    assert "ORD-1001" not in json.dumps(profile)


def test_a_category_repeats_but_an_identifier_does_not(store) -> None:
    """The rule that keeps primary keys and timestamps out of the seal.

    Both columns sit under the cardinality cap in a five-row file; only one of
    them is a category. Without the repeat test, every order id and every ISO
    date was sealed -- 21 terms for this table instead of 11.
    """

    profile = store.add("sales.csv", SALES).profile

    assert column(profile, "region")["values"] == VALUE_SEPARATOR.join(
        ["east", "north", "west"]
    )
    assert "values" not in column(profile, "order_id")
    assert "values" not in column(profile, "date")
    # The count is still reported; it is only the listing that is withheld.
    assert column(profile, "order_id")["unique"] == 5


def test_stats_come_from_a_prefix_but_the_row_count_is_exact(
    store, monkeypatch
) -> None:
    monkeypatch.setattr("resolution.attachments.MAX_ROWS_SCANNED", 3)
    rows = b"".join(b"a,%d\n" % n for n in range(50))
    profile = store.add("many.csv", b"letter,n\n" + rows).profile

    assert profile["rows"] == 50
    assert profile["stats_from_first_rows"] == 3


def test_wide_tables_are_truncated_and_say_so(store) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    width = MAX_COLUMNS + 12
    writer.writerow([f"feature_{i}" for i in range(width)])
    for row in range(4):
        writer.writerow([f"cat_{i}_{row % 2}" for i in range(width)])

    profile = store.add("wide.csv", buffer.getvalue().encode()).profile
    assert len(profile["columns"]) == MAX_COLUMNS
    assert profile["columns_omitted"] == 12


def test_duplicate_headers_do_not_collide(store) -> None:
    """Column names become dict keys downstream; duplicates would merge."""

    profile = store.add("dupes.csv", b"a,a,b\n1,2,3\n4,5,6\n").profile
    assert [c["name"] for c in profile["columns"]] == ["a", "a_1", "b"]


def test_tsv_is_read_with_its_own_delimiter(store) -> None:
    profile = store.add("t.tsv", b"left\tright\n1\t2\n").profile
    assert [c["name"] for c in profile["columns"]] == ["left", "right"]


def test_unreadable_or_unsupported_uploads_are_refused(store) -> None:
    with pytest.raises(AttachmentError):
        store.add("notes.txt", b"hello")
    with pytest.raises(AttachmentError):
        store.add("empty.csv", b"")


def test_a_header_with_no_rows_is_a_table_not_an_error(store) -> None:
    """Zero rows is a real answer about the data, not a malformed upload."""

    profile = store.add("head.csv", b"a,b\n").profile
    assert profile["rows"] == 0
    assert [c["name"] for c in profile["columns"]] == ["a", "b"]


def test_a_browser_filename_cannot_escape_the_upload_directory(store) -> None:
    item = store.add("../../etc/passwd.csv", b"a,b\n1,2\n2,3\n")
    assert item.name == "passwd.csv"
    assert item.path.parent.parent == store.root


# -- the seal --------------------------------------------------------------
def test_sealed_terms_are_the_schema_minus_what_identifies_nothing() -> None:
    terms = sealed_terms(profile_table_from(SALES))

    # Column names and repeating categories.
    assert {"region", "east", "west", "north", "channel", "gross_revenue"} <= set(terms)
    # `date` is too generic to be worth the damage sealing it does to the prose.
    assert "date" not in terms
    # Numbers and booleans contribute a name, never a value: sealing "6000.0"
    # or "true" would rewrite unrelated text throughout the statement.
    assert "true" not in terms and "false" not in terms
    assert not any(t.replace(".", "").isdigit() for t in terms)
    # Identifiers, excluded by the repeat rule.
    assert not any(t.startswith("ORD-") for t in terms)


def test_derived_terms_actually_reach_the_seal(store) -> None:
    """`native_terms` is the only reader with teeth; this is the wiring."""

    item = store.add("sales.csv", SALES)
    problem = build_problem("Why did revenue fall in the west?", attachments=[item])

    assert "west" in native_terms(problem)
    assert "region" in native_terms(problem)


def test_the_planner_never_sees_the_schema(store) -> None:
    """Anonymisation reaches into the profile, keys and category lists alike."""

    item = store.add("sales.csv", SALES)
    problem = build_problem(
        "Revenue fell in the west region on the partner channel. Why?",
        attachments=[item],
    )
    anonymized, mapping = anonymize_problem(problem)

    blob = json.dumps(anonymized.model_dump())
    for term in ("region", "west", "partner", "gross_revenue"):
        assert term not in blob, f"{term} survived anonymisation"
    # The map is GOD's alone, and reads symbol -> native term.
    assert "west" in mapping.values()
    # General vocabulary survives, or the planner cannot tell what kind of
    # problem this is.
    assert "Revenue fell" in anonymized.statement


def test_a_profile_costs_one_projection_obligation_per_column(store) -> None:
    """THE constraint that shapes the profile's schema.

    `_source_units` bills one obligation per leaf and recurses into every list
    it finds. A column that listed its categories as a list would cost one
    obligation per category, and a wide categorical table would demand
    thousands of hand-authored mappings and never validate.
    """

    item = store.add("sales.csv", SALES)
    problem = build_problem("Why?", attachments=[item])
    sources = projection_contract(problem).source_ids

    # statement + format + rows + one per column.
    assert len(sources) == 1 + 2 + len(item.profile["columns"])


def test_a_wide_categorical_table_stays_bounded(store) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    width = MAX_COLUMNS + 20
    writer.writerow([f"feature_{i}" for i in range(width)])
    for row in range(60):
        writer.writerow([f"cat_{i}_{row % 20}" for i in range(width)])

    item = store.add("wide.csv", buffer.getvalue().encode())
    problem = build_problem("Why?", attachments=[item])
    # 40 columns x 20 categories is 800 leaves if the lists nest. Flat, it is
    # statement + format + rows + columns_omitted + one per surviving column.
    assert len(projection_contract(problem).source_ids) == 1 + 3 + MAX_COLUMNS


# -- problem construction --------------------------------------------------
def test_attachments_contribute_inputs_and_entities_but_prose_never_does(
    store,
) -> None:
    item = store.add("sales.csv", SALES)
    problem = build_problem("A pump feeds a valve. Why is throughput flat?")
    # Unchanged guarantee: nothing is inferred from prose.
    assert problem.entities == []
    assert problem.inputs == {}

    attached = build_problem("Why is throughput flat?", attachments=[item])
    assert attached.inputs == {"sales.csv": item.profile}
    assert attached.entities == item.terms


def test_a_preset_keeps_its_own_entities_when_a_file_is_added(store) -> None:
    item = store.add("sales.csv", SALES)
    merged = with_attachments(toy_problem(), [item])

    assert set(toy_problem().entities) <= set(merged.entities)
    assert "region" in merged.entities
    assert merged.inputs["sales.csv"] == item.profile
    # Deduplicated and stable, so what the UI listed is what gets sealed.
    assert len(merged.entities) == len(set(merged.entities))


# -- the http surface ------------------------------------------------------
async def upload(app: ResolutionApp, name: str | None, data: bytes):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/uploads",
        "query_string": f"name={name}".encode() if name else b"",
        "headers": [(b"origin", b"http://localhost:5273")],
    }
    sent = [{"type": "http.request", "body": data, "more_body": False}]
    received: list[dict] = []

    async def receive() -> dict:
        return sent.pop(0) if sent else {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        received.append(message)

    await app(scope, receive, send)
    body = b"".join(m.get("body", b"") for m in received[1:])
    return received[0]["status"], json.loads(body)


async def start(app: ResolutionApp, payload: dict):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/runs",
        "headers": [(b"origin", b"http://localhost:5273")],
    }
    sent = [
        {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
    ]
    received: list[dict] = []

    async def receive() -> dict:
        return sent.pop(0) if sent else {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        received.append(message)

    await app(scope, receive, send)
    body = b"".join(m.get("body", b"") for m in received[1:])
    return received[0]["status"], json.loads(body)


async def test_upload_returns_the_profile_and_the_terms(tmp_path) -> None:
    app = ResolutionApp(uploads=tmp_path)
    status, body = await upload(app, "sales.csv", SALES)

    assert status == 201
    attachment = body["attachment"]
    assert attachment["name"] == "sales.csv"
    assert attachment["profile"]["rows"] == 5
    assert "region" in attachment["terms"]


async def test_bad_uploads_are_reported_not_raised(tmp_path) -> None:
    app = ResolutionApp(uploads=tmp_path)

    status, body = await upload(app, None, SALES)
    assert status == 400 and "name" in body["error"]

    status, body = await upload(app, "notes.txt", b"hello")
    assert status == 400 and ".csv" in body["error"]


async def test_an_unknown_attachment_id_is_a_400(tmp_path, monkeypatch) -> None:
    # Live support is checked before the body is, so it has to pass for the
    # attachment error to be the one under test.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    app = ResolutionApp(uploads=tmp_path)
    status, body = await start(
        app, {"mode": "live", "prompt": "why?", "attachments": ["att-nope"]}
    )
    assert status == 400
    assert "att-nope" in body["error"]


async def test_a_replay_refuses_attachments_rather_than_ignoring_them(
    tmp_path,
) -> None:
    """A recording answers its own problem whatever it is handed.

    Accepting the file would show it uploaded, mounted and silently unread.
    """

    app = ResolutionApp(uploads=tmp_path)
    _, body = await upload(app, "sales.csv", SALES)
    attachment_id = body["attachment"]["id"]

    status, body = await start(
        app,
        {
            "mode": "scripted",
            "preset": "pfk-bottleneck",
            "attachments": [attachment_id],
        },
    )
    assert status == 400
    assert "replay" in body["error"]


async def test_the_raw_file_is_seeded_and_named_for_the_sandbox(
    store, monkeypatch
) -> None:
    """The other half: the profile is projected, the bytes are mounted.

    shared/ is read-only in every sandbox, so seeding has to happen out here
    and before the first spawn -- and the runtime has to be TOLD the names, or
    `envelope_to_spec` neither lists them for the agent nor pulls pandas into
    the image.
    """

    import sys
    import types
    from unittest import mock

    from resolution import runs as runs_module

    captured: dict[str, object] = {}
    seeded: dict[str, object] = {}

    class _Runtime:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    def _seed(run_id: str, paths: list) -> list[str]:
        seeded["run_id"] = run_id
        seeded["paths"] = [str(p) for p in paths]
        return [p.name for p in paths]

    item = store.add("sales.csv", SALES)
    run = runs_module.Run(
        "run-attached",
        build_problem("Why?", attachments=[item]),
        mode="live",
        domain_count=2,
        execution=runs_module.EXECUTION_SANDBOX,
        attachments=[item],
    )

    module = types.ModuleType("reagents.demigod.sandbox_runtime")
    module.SandboxDemigodRuntime = _Runtime  # type: ignore[attr-defined]
    module.seed_shared_files = _seed  # type: ignore[attr-defined]
    session = types.ModuleType("broker.session")
    session.modal_session = lambda: object()  # type: ignore[attr-defined]
    with mock.patch.dict(
        sys.modules,
        {"reagents.demigod.sandbox_runtime": module, "broker.session": session},
    ):
        await run._build_runtime()

    assert seeded["run_id"] == "run-attached"
    assert seeded["paths"] == [str(item.path)]
    assert captured["shared_files"] == ["sales.csv"]

    # And the run says out loud that those bytes are not sealed. `build_event`
    # lowercases every kind, so this matches the stored form.
    attached = [e for e in run.events if e.kind == "attached"]
    assert len(attached) == 1
    assert "not" in attached[0].message
    assert attached[0].data["files"][0]["name"] == "sales.csv"


def profile_table_from(data: bytes):
    """`profile_table` takes a path; these fixtures are bytes."""

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sales.csv"
        path.write_bytes(data)
        return profile_table(path)
