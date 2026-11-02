"""Uretilen strateji kodu icin AST tabanli guvenlik/dogrulama katmani.

Amac: LLM kaynakli veya derlenmis kodun
    * sozdizimi hatasi icermediginden,
    * yalnizca izinli modulleri import ettiginden,
    * dosya/sistem/ag erisimi gibi tehlikeli cagrilar yapmadigindan,
    * ``BaseStrategy`` sozlesmesini (prepare/generate_signal) uyguladigindan
emin olmak.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Set

from agency.exceptions import StrategyValidationError

MAX_SOURCE_BYTES = 60_000

ALLOWED_IMPORT_ROOTS: Set[str] = {
    "__future__",
    "agency",
    "typing",
    "math",
    "statistics",
    "dataclasses",
    "numpy",
    "pandas",
}

FORBIDDEN_CALLS: Set[str] = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "getattr", "setattr", "delattr", "vars",
    "breakpoint", "exit", "quit", "memoryview", "help",
    "system", "popen", "spawn", "fork", "execv", "remove", "unlink", "rmdir",
}

FORBIDDEN_ATTRS: Set[str] = {
    "__globals__", "__dict__", "__class__", "__subclasses__", "__builtins__",
    "__code__", "__reduce__", "__mro__", "__bases__", "__loader__", "__spec__",
}

REQUIRED_METHODS = {"prepare", "generate_signal"}


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    line: int = 0

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "line": self.line,
        }


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def validate_strategy_source(source: str, *, filename: str = "<strategy>") -> List[ValidationIssue]:
    """Kaynagi denetler; bulunan sorunlarin listesini dondurur."""
    issues: List[ValidationIssue] = []
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        issues.append(ValidationIssue("error", "too_large", f"Kaynak cok buyuk ({len(encoded)} bayt)"))
        return issues

    try:
        compile(source, filename, "exec")
    except SyntaxError as exc:
        issues.append(
            ValidationIssue("error", "syntax", f"Sozdizimi hatasi: {exc.msg}", int(exc.lineno or 0))
        )
        return issues

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:  # pragma: no cover - compile zaten yakalar
        issues.append(ValidationIssue("error", "syntax", str(exc), int(exc.lineno or 0)))
        return issues

    class_names: Set[str] = set()
    base_names: Set[str] = set()
    methods: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    issues.append(
                        ValidationIssue(
                            "error", "import", f"Izin verilmeyen import: {alias.name}", node.lineno
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if node.level:
                issues.append(
                    ValidationIssue("error", "import", "Goreli import kullanilamaz", node.lineno)
                )
            elif root and root not in ALLOWED_IMPORT_ROOTS:
                issues.append(
                    ValidationIssue(
                        "error", "import", f"Izin verilmeyen import: {module}", node.lineno
                    )
                )
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FORBIDDEN_CALLS:
                issues.append(
                    ValidationIssue("error", "forbidden_call", f"Yasakli cagri: {name}()", node.lineno)
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                issues.append(
                    ValidationIssue(
                        "error",
                        "forbidden_attr",
                        f"Yasakli oznitelik erisimi: {node.attr}",
                        node.lineno,
                    )
                )
        elif isinstance(node, ast.While):
            issues.append(
                ValidationIssue("error", "while_loop", "Stratejide while dongusu yasak", node.lineno)
            )
        elif isinstance(node, ast.ClassDef):
            class_names.add(node.name)
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.add(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.add(base.attr)
            if node.name == "Strategy":
                methods = {
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                }

    if "Strategy" not in class_names:
        issues.append(ValidationIssue("error", "missing_class", "'Strategy' sinifi bulunamadi"))
    if "BaseStrategy" not in base_names:
        issues.append(
            ValidationIssue("error", "missing_base", "'Strategy' sinifi BaseStrategy'den turemeli")
        )
    missing = REQUIRED_METHODS - methods
    if missing:
        issues.append(
            ValidationIssue("error", "missing_methods", f"Eksik metodlar: {sorted(missing)}")
        )

    return issues


def assert_valid_strategy(source: str, *, filename: str = "<strategy>") -> None:
    """Sorun varsa :class:`StrategyValidationError` firlatir."""
    issues = validate_strategy_source(source, filename=filename)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise StrategyValidationError(
            "Strateji kodu dogrulamadan gecemedi: " + "; ".join(e.message for e in errors[:5]),
            details={"issues": [e.to_dict() for e in issues], "filename": filename},
        )

