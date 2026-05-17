from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_TARGETS = (
    REPO_ROOT / "src",
    REPO_ROOT / "environment.yml",
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "LICENSE",
)
TEXT_FILE_SUFFIXES = {".py", ".toml", ".yaml", ".yml"}
FORBIDDEN_PATTERNS = (
    "dynamicprot.src",
    "vendored",
    "submit_vendored_eval_suite",
    "install_dprot_env",
    "sys.path.append",
    "package://examples/eval/pdb",
    "/rds/",
    "/home/",
    "/lus/",
)


def _iter_scan_files():
    for target in SCAN_TARGETS:
        if target.is_dir():
            yield from sorted(
                path
                for path in target.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in TEXT_FILE_SUFFIXES
            )
        elif target.is_file():
            yield target


def test_publication_surface_has_no_stale_repo_specific_references():
    violations: list[str] = []

    for path in _iter_scan_files():
        content = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(REPO_ROOT)

        if "dynamicprot" in content:
            violations.append(f"{relative_path}: dynamicprot")

        for pattern in FORBIDDEN_PATTERNS:
            if pattern in content:
                violations.append(f"{relative_path}: {pattern}")

    assert not violations, "\n".join(violations)
