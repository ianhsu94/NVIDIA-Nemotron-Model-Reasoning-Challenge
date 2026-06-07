from __future__ import annotations

from teacher_training.ranker import rank_symbol
from teacher_training.solvers.symbol_solvers import all_symbol_solvers
from teacher_training.teacher_base import BaseTeacher, TeacherResult


class SymbolTeacher(BaseTeacher):
    name = "Teacher-Symbol"
    category = "symbol"

    def __init__(self, solver_prior: dict[str, float] | None = None):
        self.solver_prior = solver_prior or {}
        self.solvers = all_symbol_solvers()

    def predict_row(self, row) -> TeacherResult:
        results = [solver.solve(row) for solver in self.solvers]
        best = rank_symbol(results, self.solver_prior)
        return TeacherResult(
            answer=str(best.get("answer", "")),
            confidence=float(best.get("confidence", 0.0)),
            selected_solver=str(best.get("solver_name", "")),
            matched_examples=int(best.get("matched_examples", 0)),
            total_examples=int(best.get("total_examples", 0)),
            reason=str(best.get("reason", "")),
        )

