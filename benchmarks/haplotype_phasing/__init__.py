"""Real-data long-read haplotype phasing benchmark."""

from benchmarks.haplotype_phasing.benchmark import (
    HaplotypePhasingVerifier,
    load_problem,
    score_solution,
)

__all__ = ["HaplotypePhasingVerifier", "load_problem", "score_solution"]
