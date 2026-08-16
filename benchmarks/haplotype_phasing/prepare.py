"""Derive and freeze the HG004 benchmark from the official WhatsHap fixture.

This script is the only benchmark code that needs pysam. Runtime tools consume
the generated public module; private expected phase never enters that module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
CASE_DIR = Path(__file__).parent
PUBLIC_DIR = CASE_DIR / "public"
PRIVATE_DIR = CASE_DIR / "private"
TOOL_DATA = ROOT / "src" / "reagents" / "tools" / "_haplotype_public.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variants_from_vcf(path: Path) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            chrom, pos, _, ref, alt = fields[:5]
            genotype = fields[9].split(":", 1)[0]
            if (
                len(ref) != 1
                or len(alt) != 1
                or "," in alt
                or genotype not in {"0/1", "1/0", "0|1", "1|0"}
            ):
                continue
            variants.append(
                {
                    "variant_id": f"V{len(variants) + 1:03d}",
                    "contig": chrom,
                    "fixture_position": int(pos),
                    "reference": ref,
                    "alternate": alt,
                }
            )
    return variants


def public_matrix(vcf_path: Path, bam_path: Path) -> dict[str, Any]:
    try:
        import pysam
    except ModuleNotFoundError as exc:  # pragma: no cover - setup guidance
        raise SystemExit(
            "prepare.py requires pysam (install the biology extra)"
        ) from exc

    variants = variants_from_vcf(vcf_path)
    by_zero_pos = {item["fixture_position"] - 1: item for item in variants}
    reads: list[dict[str, Any]] = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for record in bam.fetch(until_eof=True):
            if (
                record.is_unmapped
                or record.is_secondary
                or record.is_supplementary
                or record.is_duplicate
                or record.query_sequence is None
            ):
                continue
            calls: list[dict[str, Any]] = []
            qualities = record.query_qualities
            for query_pos, ref_pos in record.get_aligned_pairs(matches_only=True):
                if query_pos is None or ref_pos not in by_zero_pos:
                    continue
                variant = by_zero_pos[ref_pos]
                base = record.query_sequence[query_pos].upper()
                if base == variant["reference"]:
                    allele = 0
                elif base == variant["alternate"]:
                    allele = 1
                else:
                    continue
                quality = int(qualities[query_pos]) if qualities else 10
                calls.append(
                    {
                        "variant_id": variant["variant_id"],
                        "allele": allele,
                        "quality": max(1, min(40, quality)),
                    }
                )
            if len(calls) < 2:
                continue
            opaque = hashlib.sha256(record.query_name.encode()).hexdigest()[:12]
            reads.append(
                {
                    "read_id": f"R{len(reads) + 1:03d}-{opaque}",
                    "mapping_quality": int(record.mapping_quality),
                    "calls": sorted(calls, key=lambda item: item["variant_id"]),
                }
            )
    covered = {call["variant_id"] for read in reads for call in read["calls"]}
    variants = [item for item in variants if item["variant_id"] in covered]
    return {
        "schema_version": "1.0",
        "benchmark_id": "giab-hg004-long-read-phasing-v1",
        "provenance": {
            "source_project": "WhatsHap",
            "source_sample": "GIAB HG004 / NA24143",
            "source_region": "GRCh37 chromosome 6 source interval 10029001-10055081",
            "source_fixture": "tests/data/pacbio",
            "license": "MIT for WhatsHap fixture; underlying GIAB data are public",
        },
        "allele_encoding": {"0": "reference", "1": "alternate"},
        "variants": variants,
        "reads": reads,
    }


def reference_phase(public: dict[str, Any], phased_vcf: Path) -> dict[str, Any]:
    by_key = {
        (
            item["contig"],
            item["fixture_position"],
            item["reference"],
            item["alternate"],
        ): item["variant_id"]
        for item in public["variants"]
    }
    phase: dict[str, int] = {}
    phase_sets: dict[str, str | None] = {}
    with phased_vcf.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            chrom, pos, _, ref, alt = fields[:5]
            variant_id = by_key.get((chrom, int(pos), ref, alt))
            if variant_id is None:
                continue
            sample = dict(zip(fields[8].split(":"), fields[9].split(":"), strict=False))
            genotype = sample.get("GT", "")
            if genotype not in {"0|1", "1|0"}:
                continue
            phase[variant_id] = int(genotype.split("|")[0])
            phase_sets[variant_id] = sample.get("PS")
    return {
        "reference_kind": "official WhatsHap HG004 fixture output",
        "orientation_invariant": True,
        "phase": phase,
        "phase_sets": phase_sets,
    }


def weighted_discordance(public: dict[str, Any], phase: dict[str, int]) -> int:
    total = 0
    for read in public["reads"]:
        direct = sum(
            call["quality"]
            for call in read["calls"]
            if phase[call["variant_id"]] != call["allele"]
        )
        complement = sum(call["quality"] for call in read["calls"]) - direct
        total += min(direct, complement)
    return total


def exact_optimum(public: dict[str, Any]) -> tuple[dict[str, int], int]:
    try:
        import z3
    except ModuleNotFoundError as exc:  # pragma: no cover - setup guidance
        raise SystemExit(
            "prepare.py requires z3-solver (install reasoning extra)"
        ) from exc

    variant_ids = [item["variant_id"] for item in public["variants"]]
    phase = {variant_id: z3.Bool(f"x_{variant_id}") for variant_id in variant_ids}
    orientations = [
        z3.Bool(f"read_{index:03d}") for index in range(len(public["reads"]))
    ]
    penalties = []
    for orientation, read in zip(orientations, public["reads"], strict=True):
        for call in read["calls"]:
            observed = z3.BoolVal(bool(call["allele"]))
            expected = z3.Xor(phase[call["variant_id"]], orientation)
            penalties.append(z3.If(expected == observed, 0, int(call["quality"])))
    optimizer = z3.Optimize()
    optimizer.add(phase[variant_ids[0]] == z3.BoolVal(False))
    objective = optimizer.minimize(z3.Sum(penalties))
    if optimizer.check() != z3.sat:
        raise RuntimeError("weighted phase objective was unexpectedly unsatisfiable")
    model = optimizer.model()
    assignment = {
        variant_id: int(
            z3.is_true(model.eval(phase[variant_id], model_completion=True))
        )
        for variant_id in variant_ids
    }
    cost = int(str(objective.value()))
    assert weighted_discordance(public, assignment) == cost
    return assignment, cost


def write_fixture(source_dir: Path) -> dict[str, Any]:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    public = public_matrix(source_dir / "variants.vcf", source_dir / "pacbio.bam")
    expected = reference_phase(public, source_dir / "phased.vcf")
    optimum, optimum_cost = exact_optimum(public)
    expected["exact_public_objective_optimum"] = {
        "phase": optimum,
        "weighted_discordance": optimum_cost,
    }

    observations = PUBLIC_DIR / "observations.json"
    private = PRIVATE_DIR / "expected.json"
    observations.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    private.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    TOOL_DATA.write_text(
        '"""Generated public HG004 evidence. Do not edit by hand."""\n\n'
        f"DATA = {public!r}\n",
        encoding="utf-8",
    )
    question = PUBLIC_DIR / "question.json"
    manifest = {
        "benchmark_id": public["benchmark_id"],
        "frozen_before_model_calls": True,
        "heldout_reference_available_to_agents": False,
        "public_question_sha256": _sha(question),
        "public_observations_sha256": _sha(observations),
        "source_files": {
            name: _sha(source_dir / name)
            for name in ("variants.vcf", "pacbio.bam", "phased.vcf")
        },
        "counts": {
            "variants": len(public["variants"]),
            "reads": len(public["reads"]),
            "allele_calls": sum(len(read["calls"]) for read in public["reads"]),
        },
    }
    (PUBLIC_DIR / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (PRIVATE_DIR / "freeze_seal.json").write_text(
        json.dumps(
            {
                "benchmark_id": public["benchmark_id"],
                "private_expected_sha256": _sha(private),
                "frozen_before_model_calls": True,
                "disclosed_to_agents": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to whatshap/tests/data/pacbio from an official checkout.",
    )
    args = parser.parse_args()
    print(json.dumps(write_fixture(args.source_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
