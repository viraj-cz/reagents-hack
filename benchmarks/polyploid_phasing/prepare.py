"""Freeze the real-read tetraploid benchmark from the WhatsHap fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
CASE_DIR = Path(__file__).parent
PUBLIC_DIR = CASE_DIR / "public"
PRIVATE_DIR = CASE_DIR / "private"
TOOL_DATA = ROOT / "src" / "reagents" / "tools" / "_polyploid_public.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_variants(path: Path) -> list[dict[str, Any]]:
    variants = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            chrom, pos, _, ref, alt = fields[:5]
            genotype = fields[9].split(":", 1)[0].replace("|", "/")
            alleles = [int(value) for value in genotype.split("/")]
            if len(alleles) != 4 or len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            variants.append(
                {
                    "variant_id": f"V{len(variants) + 1:03d}",
                    "contig": chrom,
                    "position": int(pos),
                    "reference": ref,
                    "alternate": alt,
                    "alternate_dosage": sum(alleles),
                }
            )
    return variants


def public_matrix(vcf: Path, bam: Path) -> dict[str, Any]:
    import pysam

    variants = parse_variants(vcf)
    by_pos = {item["position"] - 1: item for item in variants}
    reads = []
    with pysam.AlignmentFile(str(bam), "rb") as alignments:
        for record in alignments.fetch(until_eof=True):
            if record.is_unmapped or record.query_sequence is None:
                continue
            calls = []
            qualities = record.query_qualities
            for query_pos, ref_pos in record.get_aligned_pairs(matches_only=True):
                if query_pos is None or ref_pos not in by_pos:
                    continue
                variant = by_pos[ref_pos]
                base = record.query_sequence[query_pos].upper()
                if base == variant["reference"]:
                    bit = 0
                elif base == variant["alternate"]:
                    bit = 1
                else:
                    continue
                quality = int(qualities[query_pos]) if qualities else 10
                calls.append(
                    {
                        "variant_id": variant["variant_id"],
                        "bit": bit,
                        "weight": max(1, min(40, quality)),
                    }
                )
            if len(calls) < 2:
                continue
            opaque = hashlib.sha256(record.query_name.encode()).hexdigest()[:12]
            reads.append(
                {
                    "read_id": f"R{len(reads) + 1:03d}-{opaque}",
                    "calls": sorted(calls, key=lambda item: item["variant_id"]),
                }
            )
    return {
        "schema_version": "1.0",
        "benchmark_id": "real-human-tetraploid-long-read-phasing-v1",
        "ploidy": 4,
        "variants": variants,
        "reads": reads,
        "provenance": {
            "source": "WhatsHap tests/data/polyploid.chr22.42M.12k",
            "samples": ["HG00514", "NA19240"],
            "evidence": "real PacBio reads pooled as an artificial tetraploid",
        },
    }


def run_reference(whatshap: Path, vcf: Path, bam: Path, output: Path) -> None:
    subprocess.run(
        [
            str(whatshap),
            "polyphase",
            "--ploidy",
            "4",
            "--ignore-read-groups",
            "-o",
            str(output),
            str(vcf),
            str(bam),
        ],
        check=True,
    )


def parse_reference(public: dict[str, Any], path: Path) -> dict[str, Any]:
    by_key = {
        (item["contig"], item["position"], item["reference"], item["alternate"]): item[
            "variant_id"
        ]
        for item in public["variants"]
    }
    phased = {}
    blocks = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            variant_id = by_key[(fields[0], int(fields[1]), fields[3], fields[4])]
            sample = dict(zip(fields[8].split(":"), fields[9].split(":"), strict=False))
            genotype = sample.get("GT", "")
            if "|" in genotype:
                phased[variant_id] = [int(value) for value in genotype.split("|")]
                blocks[variant_id] = sample.get("PS")
    return {
        "reference_kind": "WhatsHap 2.8 polyphase output",
        "haplotype_labels_exchangeable": True,
        "phased": phased,
        "phase_set": blocks,
    }


def objective(public: dict[str, Any], haplotypes: list[dict[str, int]]) -> int:
    total = 0
    for read in public["reads"]:
        costs = [
            sum(
                call["weight"]
                for call in read["calls"]
                if haplotype[call["variant_id"]] != call["bit"]
            )
            for haplotype in haplotypes
        ]
        total += min(costs)
    return total


def exact_optimum(public: dict[str, Any]) -> tuple[list[dict[str, int]], int]:
    import z3

    ids = [item["variant_id"] for item in public["variants"]]
    x = [[z3.Bool(f"h{h}_{variant}") for variant in ids] for h in range(4)]
    index = {variant: i for i, variant in enumerate(ids)}
    solver = z3.Optimize()
    for column, variant in enumerate(public["variants"]):
        solver.add(
            z3.Sum([z3.If(x[row][column], 1, 0) for row in range(4)])
            == variant["alternate_dosage"]
        )
    clusters = [z3.Int(f"cluster_{i}") for i in range(len(public["reads"]))]
    penalties = []
    for cluster, read in zip(clusters, public["reads"], strict=True):
        solver.add(cluster >= 0, cluster < 4)
        for call in read["calls"]:
            observed = z3.BoolVal(bool(call["bit"]))
            selected = z3.Or(
                [
                    z3.And(
                        cluster == row, x[row][index[call["variant_id"]]] == observed
                    )
                    for row in range(4)
                ]
            )
            penalties.append(z3.If(selected, 0, call["weight"]))
    handle = solver.minimize(z3.Sum(penalties))
    if solver.check() != z3.sat:
        raise RuntimeError("tetraploid objective is unsatisfiable")
    model = solver.model()
    haplotypes = [
        {
            variant: int(z3.is_true(model.eval(x[row][column], model_completion=True)))
            for column, variant in enumerate(ids)
        }
        for row in range(4)
    ]
    cost = int(str(handle.value()))
    assert objective(public, haplotypes) == cost
    return haplotypes, cost


def write_fixture(source_dir: Path, whatshap: Path) -> dict[str, Any]:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    vcf = source_dir / "polyploid.chr22.42M.12k.vcf"
    bam = source_dir / "polyploid.chr22.42M.12k.bam"
    public = public_matrix(vcf, bam)
    reference_vcf = PRIVATE_DIR / "whatshap-reference.vcf"
    run_reference(whatshap, vcf, bam, reference_vcf)
    expected = parse_reference(public, reference_vcf)
    optimum, cost = exact_optimum(public)
    expected["exact_public_objective_optimum"] = {
        "haplotypes": optimum,
        "weighted_discordance": cost,
    }
    # WhatsHap labels are exchangeable independently across its phase sets.
    # Use the exact optimum only to fill unsupported columns, then overlay the
    # official phased columns. Scoring aligns rows independently per block.
    canonical = [dict(row) for row in optimum]
    for variant_id, column in expected["phased"].items():
        for row in range(4):
            canonical[row][variant_id] = column[row]
    expected["canonical_reference"] = {
        "haplotypes": canonical,
        "weighted_discordance": objective(public, canonical),
    }
    observations = PUBLIC_DIR / "observations.json"
    private = PRIVATE_DIR / "expected.json"
    observations.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    private.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    TOOL_DATA.write_text(
        '"""Generated public tetraploid evidence. Do not edit."""\n\n'
        f"DATA = {public!r}\n",
        encoding="utf-8",
    )
    manifest = {
        "benchmark_id": public["benchmark_id"],
        "frozen_before_model_calls": True,
        "heldout_reference_available_to_agents": False,
        "public_question_sha256": _sha(PUBLIC_DIR / "question.json"),
        "public_observations_sha256": _sha(observations),
        "source_files": {"vcf": _sha(vcf), "bam": _sha(bam)},
        "counts": {
            "ploidy": 4,
            "variants": len(public["variants"]),
            "informative_reads": len(public["reads"]),
            "calls": sum(len(read["calls"]) for read in public["reads"]),
        },
    }
    (PUBLIC_DIR / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (PRIVATE_DIR / "freeze_seal.json").write_text(
        json.dumps(
            {
                "private_expected_sha256": _sha(private),
                "reference_vcf_sha256": _sha(reference_vcf),
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
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--whatshap", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_fixture(args.source_dir, args.whatshap), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
