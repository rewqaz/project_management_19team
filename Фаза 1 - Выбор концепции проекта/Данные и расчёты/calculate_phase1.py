"""Reproduce Phase 1's illustrative TOPSIS and group AHP calculations.

All four profiles in model_inputs.json are simulated. No values are presented
as independent judgments made by the students.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "model_inputs.json").read_text(encoding="utf-8"))
EXPERTS = DATA["experts"]
ALT = ["A1", "A2", "A3", "A4"]
CRIT = ["C1", "C2", "C3", "C4"]
RI4 = 0.90


def triple(expected: int) -> list[int]:
    return [max(1, expected - 1), expected, min(9, expected + 1)]


def mean_triples(rows: list[list[int]]) -> list[float]:
    return [sum(row[k] for row in rows) / len(rows) for k in range(3)]


def vertex(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)) / 3)


SAATY = [1 / n for n in range(9, 1, -1)] + list(range(1, 10))


def saaty_ratio(first: float, second: float) -> float:
    ratio = first / second
    return min(SAATY, key=lambda x: (abs(math.log(x / ratio)), abs(x - ratio)))


def pair_matrix(seed: list[float]) -> list[list[float]]:
    matrix = [[1.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            matrix[i][j] = saaty_ratio(seed[i], seed[j])
            matrix[j][i] = 1 / matrix[i][j]
    return matrix


def group_matrix(matrices: list[list[list[float]]]) -> list[list[float]]:
    n = len(matrices)
    return [[math.prod(matrix[i][j] for matrix in matrices) ** (1 / n)
             for j in range(4)] for i in range(4)]


def ahp_priority(matrix: list[list[float]]) -> tuple[list[float], dict[str, float]]:
    row_gm = [math.prod(row) ** (1 / 4) for row in matrix]
    total = sum(row_gm)
    weights = [value / total for value in row_gm]
    aw = [sum(matrix[i][j] * weights[j] for j in range(4)) for i in range(4)]
    lam = sum(aw[i] / weights[i] for i in range(4)) / 4
    ci = (lam - 4) / 3
    return weights, {"lambda_max": lam, "CI": ci, "CR": ci / RI4}


def compute(experts: list[dict]) -> dict:
    n = len(experts)
    ew = [[triple(m) for m in e["criterion_expected"]] for e in experts]
    er = [[[triple(m) for m in row] for row in e["alternative_expected"]]
          for e in experts]
    weights = [mean_triples([ew[e][j] for e in range(n)]) for j in range(4)]
    ratings = [[mean_triples([er[e][i][j] for e in range(n)]) for j in range(4)]
               for i in range(4)]
    divisor = [max(ratings[i][j][2] for i in range(4)) for j in range(4)]
    weight_divisor = sum(w[2] for w in weights)
    normalized_weights = [[x / weight_divisor for x in w] for w in weights]
    normalized = [[[x / divisor[j] for x in ratings[i][j]]
                   for j in range(4)] for i in range(4)]
    weighted = [[[normalized[i][j][k] * normalized_weights[j][k]
                  for k in range(3)] for j in range(4)] for i in range(4)]
    positive = [[max(weighted[i][j][k] for i in range(4)) for k in range(3)]
                for j in range(4)]
    negative = [[min(weighted[i][j][k] for i in range(4)) for k in range(3)]
                for j in range(4)]
    distance_plus = [sum(vertex(weighted[i][j], positive[j]) for j in range(4))
                     for i in range(4)]
    distance_minus = [sum(vertex(weighted[i][j], negative[j]) for j in range(4))
                      for i in range(4)]
    closeness = [distance_minus[i] / (distance_plus[i] + distance_minus[i])
                 for i in range(4)]

    individual = []
    for e in experts:
        matrices = [pair_matrix(e["ahp_criterion_seed"])] + [
            pair_matrix(row) for row in e["ahp_alternative_seeds"]]
        individual.append(matrices)
    grouped = [group_matrix([individual[e][m] for e in range(n)]) for m in range(5)]
    priorities = [ahp_priority(matrix) for matrix in grouped]
    criterion_priority = priorities[0][0]
    local_priority = [priorities[k + 1][0] for k in range(4)]
    global_priority = [sum(criterion_priority[j] * local_priority[j][i]
                           for j in range(4)) for i in range(4)]
    individual_cr = [
        [ahp_priority(matrix)[1]["CR"] for matrix in matrices]
        for matrices in individual
    ]
    return {
        "expert_count": n,
        "fuzzy": {
            "group_weights": weights,
            "weight_divisor": weight_divisor,
            "normalized_weights": normalized_weights,
            "group_ratings": ratings,
            "rating_divisors": divisor,
            "normalized_ratings": normalized,
            "weighted_ratings": weighted,
            "positive_ideal": positive,
            "negative_ideal": negative,
            "distance_positive": distance_plus,
            "distance_negative": distance_minus,
            "closeness": closeness,
            "ranking": sorted(ALT, key=lambda a: -closeness[ALT.index(a)]),
        },
        "ahp": {
            "individual_matrices": individual,
            "group_matrices": grouped,
            "individual_cr": individual_cr,
            "criterion_priority": criterion_priority,
            "criterion_consistency": priorities[0][1],
            "local_priorities": local_priority,
            "local_consistency": [p[1] for p in priorities[1:]],
            "global_priority": global_priority,
            "ranking": sorted(ALT, key=lambda a: -global_priority[ALT.index(a)]),
        },
    }


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    results = compute(EXPERTS)
    # The separate questionnaire is a readable projection of these exact inputs.
    expanded = []
    for e in EXPERTS:
        pairs = [pair_matrix(e["ahp_criterion_seed"])] + [
            pair_matrix(row) for row in e["ahp_alternative_seeds"]]
        expanded.append({
            "id": e["id"], "profile": e["profile"],
            "criterion_triples": [triple(x) for x in e["criterion_expected"]],
            "alternative_triples": [[triple(x) for x in row]
                                    for row in e["alternative_expected"]],
            "pairwise_matrices": pairs,
        })
    (ROOT / "expanded_inputs.json").write_text(
        json.dumps({"provenance": DATA["note"], "experts": expanded},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    f = results["fuzzy"]
    write_csv(ROOT / "topsis_summary.csv",
              ["concept", "D_positive", "D_negative", "CC", "rank"],
              [[ALT[i], f["distance_positive"][i], f["distance_negative"][i],
                f["closeness"][i], f["ranking"].index(ALT[i]) + 1]
               for i in range(4)])
    a = results["ahp"]
    write_csv(ROOT / "ahp_summary.csv",
              ["concept", "C1", "C2", "C3", "C4", "global_priority", "rank"],
              [[ALT[i]] + [a["local_priorities"][j][i] for j in range(4)]
               + [a["global_priority"][i], a["ranking"].index(ALT[i]) + 1]
               for i in range(4)])
    loo = []
    for omitted in range(4):
        subset = [e for k, e in enumerate(EXPERTS) if k != omitted]
        r = compute(subset)
        loo.append({"excluded": EXPERTS[omitted]["id"],
                    "TOPSIS": r["fuzzy"]["ranking"], "AHP": r["ahp"]["ranking"]})
    (ROOT / "sensitivity.json").write_text(
        json.dumps(loo, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOPSIS", [(ALT[i], round(f["closeness"][i], 6)) for i in range(4)])
    print("AHP", [(ALT[i], round(a["global_priority"][i], 6)) for i in range(4)])
    print("group CR", [round(a["criterion_consistency"]["CR"], 5)] +
          [round(x["CR"], 5) for x in a["local_consistency"]])
    print("individual CR", [[round(x, 5) for x in row] for row in a["individual_cr"]])
    print("LOO", loo)


if __name__ == "__main__":
    main()
