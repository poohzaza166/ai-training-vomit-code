"""Training project families for project_env: plain-language outlines, hidden checks, reference solutions.

None of these overlap the real-use evaluation (no dice, music, Discord, chat bots or queues). Each
family is parameterised by a seed so names, wording and data vary between episodes. A task carries:

  outline       what the user sends: plain language, plus a short "how people will use it" section so
                behaviour can be checked automatically (the evaluation spec has no such section)
  package       the command name the outline asks for (`python3 -m <package>`)
  keywords      requirement words a real spec draft is expected to cover
  checks        hidden CLI checks: (args, expected regexes in stdout, should_succeed)
  reference()   a reference solution: spec draft, modules, tests, owner guide (for SFT demonstrations)
"""
from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass, field


@dataclass
class Check:
    args: list[str]
    expect: list[str] = field(default_factory=list)   # regexes that must appear in stdout
    ok: bool = True                                    # False: must fail politely (non-zero exit, no traceback)


@dataclass
class Task:
    family: str
    slug: str
    package: str
    outline: str
    keywords: list[str]
    checks: list[Check]
    files: dict[str, str]          # reference solution, relative path -> content
    spec_doc: str                  # reference spec draft (written first)
    readme: str

    def reference(self):
        return self.spec_doc, self.files, self.readme


def _d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def _pick(rng, *options):
    return rng.choice(options)


# --------------------------------------------------------------------------------------------------
# 1. Expense ledger
# --------------------------------------------------------------------------------------------------

def ledger(rng: random.Random) -> Task:
    pkg = _pick(rng, "ledger", "spendlog", "budgetbook", "expenses")
    who = _pick(rng, "a student sharing a flat", "a freelancer", "a family", "a small club treasurer")
    cur = _pick(rng, "$", "€", "£")
    outline = _d(f"""
        I want a small command-line expense tracker for {who}. Put everything for it in this folder.

        What it should do
        - Record an expense with an amount, a category and an optional note.
        - List all recorded expenses in the order they were added.
        - Show the total spent, either overall or for one category.
        - Show a summary with the total for each category.
        - Reject amounts that are not positive numbers with a clear message instead of crashing.
        - Remember expenses between runs.

        How people will use it
        - `python3 -m {pkg} add 12.50 food "lunch"` records an expense and prints `added`.
        - `python3 -m {pkg} total` prints `total: {cur}15.50` (two decimals).
        - `python3 -m {pkg} total --category food` prints the total for that category the same way.
        - `python3 -m {pkg} summary` prints one line per category like `food: {cur}12.50`.
        - `python3 -m {pkg} list` prints one line per expense with its amount, category and note.

        Quality expectations
        - Keep it organized so someone else can maintain it, with a short guide for the owner.
        - Include automated checks that prove it works, and make sure they pass before you finish.
        """)
    storage = _d(f'''
        """Persistent storage for expenses as a JSON file."""
        from __future__ import annotations

        import json
        import os
        from pathlib import Path

        from .models import Expense

        DEFAULT_FILE = ".{pkg}.json"


        def data_path() -> Path:
            """Where expenses are stored: ${pkg.upper()}_FILE if set, else a file in the current folder."""
            return Path(os.environ.get("{pkg.upper()}_FILE", DEFAULT_FILE))


        def load(path: Path | None = None) -> list[Expense]:
            path = path or data_path()
            if not path.exists():
                return []
            return [Expense(**row) for row in json.loads(path.read_text())]


        def save(expenses: list[Expense], path: Path | None = None) -> None:
            path = path or data_path()
            path.write_text(json.dumps([e.__dict__ for e in expenses], indent=2))
        ''')
    models = _d('''
        """Expense records and the calculations on them."""
        from __future__ import annotations

        from dataclasses import dataclass


        @dataclass
        class Expense:
            amount: float
            category: str
            note: str = ""


        def parse_amount(text: str) -> float:
            """Return a positive amount, or raise ValueError with a readable message."""
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"amount must be a number, got {text!r}") from None
            if value <= 0:
                raise ValueError("amount must be greater than zero")
            return round(value, 2)


        def total(expenses: list[Expense], category: str | None = None) -> float:
            return round(sum(e.amount for e in expenses if category in (None, e.category)), 2)


        def by_category(expenses: list[Expense]) -> dict[str, float]:
            sums: dict[str, float] = {}
            for e in expenses:
                sums[e.category] = round(sums.get(e.category, 0.0) + e.amount, 2)
            return dict(sorted(sums.items()))
        ''')
    cli = _d(f'''
        """Command-line interface: argument parsing and output only; logic lives in models."""
        from __future__ import annotations

        import argparse
        import sys

        from . import storage
        from .models import Expense, by_category, parse_amount, total

        CURRENCY = "{cur}"


        def money(value: float) -> str:
            return f"{{CURRENCY}}{{value:.2f}}"


        def build_parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(prog="{pkg}", description="Track expenses.")
            sub = parser.add_subparsers(dest="command", required=True)
            add = sub.add_parser("add", help="record an expense")
            add.add_argument("amount")
            add.add_argument("category")
            add.add_argument("note", nargs="?", default="")
            tot = sub.add_parser("total", help="show the total spent")
            tot.add_argument("--category")
            sub.add_parser("summary", help="total per category")
            sub.add_parser("list", help="list expenses")
            return parser


        def main(argv: list[str] | None = None) -> int:
            args = build_parser().parse_args(argv)
            expenses = storage.load()
            if args.command == "add":
                try:
                    amount = parse_amount(args.amount)
                except ValueError as exc:
                    print(f"error: {{exc}}", file=sys.stderr)
                    return 2
                expenses.append(Expense(amount, args.category, args.note))
                storage.save(expenses)
                print("added")
            elif args.command == "total":
                print(f"total: {{money(total(expenses, args.category))}}")
            elif args.command == "summary":
                for category, value in by_category(expenses).items():
                    print(f"{{category}}: {{money(value)}}")
            else:
                for e in expenses:
                    print(f"{{money(e.amount)}}  {{e.category}}  {{e.note}}".rstrip())
            return 0
        ''')
    tests_models = _d(f'''
        import pytest

        from {pkg}.models import Expense, by_category, parse_amount, total


        def test_parse_amount_accepts_positive_numbers():
            assert parse_amount("12.5") == 12.5


        @pytest.mark.parametrize("text", ["abc", "0", "-3"])
        def test_parse_amount_rejects_bad_input(text):
            with pytest.raises(ValueError):
                parse_amount(text)


        def test_totals_overall_and_by_category():
            items = [Expense(12.5, "food"), Expense(3, "transport"), Expense(2.25, "food")]
            assert total(items) == 17.75
            assert total(items, "food") == 14.75
            assert by_category(items) == {{"food": 14.75, "transport": 3.0}}
        ''')
    tests_cli = _d(f'''
        from {pkg}.cli import main


        def test_add_total_and_summary(tmp_path, monkeypatch, capsys):
            monkeypatch.setenv("{pkg.upper()}_FILE", str(tmp_path / "data.json"))
            assert main(["add", "12.50", "food", "lunch"]) == 0
            assert main(["add", "3", "transport"]) == 0
            capsys.readouterr()
            main(["total"])
            assert capsys.readouterr().out.strip() == "total: {cur}15.50"
            main(["summary"])
            assert "food: {cur}12.50" in capsys.readouterr().out


        def test_bad_amount_is_rejected(tmp_path, monkeypatch, capsys):
            monkeypatch.setenv("{pkg.upper()}_FILE", str(tmp_path / "data.json"))
            assert main(["add", "abc", "food"]) == 2
            assert "error" in capsys.readouterr().err
        ''')
    files = {f"{pkg}/__init__.py": f'"""{pkg}: a small command-line expense tracker."""\n',
             f"{pkg}/__main__.py": "from .cli import main\n\nraise SystemExit(main())\n",
             f"{pkg}/models.py": models, f"{pkg}/storage.py": storage, f"{pkg}/cli.py": cli,
             "tests/test_models.py": tests_models, "tests/test_cli.py": tests_cli}
    spec = _d(f"""
        # Spec: {pkg} expense tracker

        ## Goal
        A command-line tool for {who} to record expenses and see where money goes.

        ## Requirements
        - Record an expense: amount, category, optional note; print `added`.
        - Reject non-numeric, zero or negative amounts with an error message and exit code 2 (no crash).
        - List expenses in the order added.
        - Total overall or for one category, printed as `total: {cur}X.XX`.
        - Summary: one line per category, `category: {cur}X.XX`, sorted by category.
        - Expenses persist between runs in a JSON file.

        ## Design
        - `{pkg}/models.py`: `Expense` dataclass, amount parsing/validation, totals (pure functions).
        - `{pkg}/storage.py`: load/save JSON; file path from `{pkg.upper()}_FILE` or `.{pkg}.json`.
        - `{pkg}/cli.py`: argparse commands `add`, `list`, `total`, `summary`; output formatting only.
        - `{pkg}/__main__.py`: entry point for `python3 -m {pkg}`.

        ## Verification
        - pytest unit tests for parsing and totals; CLI tests using a temporary data file.
        - Run the tests in a virtualenv (`.venv`) before finishing.
        """)
    readme = _d(f"""
        # {pkg}

        A small command-line expense tracker.

        ## Setup
        ```
        python3 -m venv .venv
        .venv/bin/pip install -r requirements.txt
        ```

        ## Usage
        ```
        python3 -m {pkg} add 12.50 food "lunch"
        python3 -m {pkg} list
        python3 -m {pkg} total [--category food]
        python3 -m {pkg} summary
        ```
        Data is stored in `.{pkg}.json` in the current folder (override with `{pkg.upper()}_FILE`).

        ## Running the tests
        ```
        .venv/bin/python -m pytest -q
        ```

        ## Layout
        - `{pkg}/models.py`: expense records and calculations
        - `{pkg}/storage.py`: saving and loading
        - `{pkg}/cli.py`: the commands
        - `tests/`: automated checks
        """)
    files["requirements.txt"] = "pytest\n"
    checks = [Check(["add", "12.50", "food", "lunch"], [r"added"]),
              Check(["add", "3", "transport"], [r"added"]),
              Check(["total"], [rf"total:\s*{_re(cur)}15\.50"]),
              Check(["total", "--category", "food"], [rf"total:\s*{_re(cur)}12\.50"]),
              Check(["summary"], [rf"food:\s*{_re(cur)}12\.50", rf"transport:\s*{_re(cur)}3\.00"]),
              Check(["list"], [r"lunch"]),
              Check(["add", "abc", "food"], [], ok=False)]
    return Task("ledger", f"{pkg}-tracker", pkg, outline,
                ["expense", "category", "total", "summary", "note", "persist", "test"], checks,
                files, spec, readme)


def _re(s: str) -> str:
    import re
    return re.escape(s)


def _common_files(pkg: str, about: str) -> dict[str, str]:
    return {f"{pkg}/__init__.py": f'"""{pkg}: {about}"""\n',
            f"{pkg}/__main__.py": "from .cli import main\n\nraise SystemExit(main())\n",
            "requirements.txt": "pytest\n"}


def _readme(pkg: str, title: str, usage: list[str], layout: dict[str, str], extra: str = "") -> str:
    lines = [f"# {pkg}", "", title, "", "## Setup", "```", "python3 -m venv .venv",
             ".venv/bin/pip install -r requirements.txt", "```", "", "## Usage", "```", *usage, "```"]
    if extra:
        lines += ["", extra]
    lines += ["", "## Running the tests", "```", ".venv/bin/python -m pytest -q", "```", "", "## Layout"]
    lines += [f"- `{k}`: {v}" for k, v in layout.items()] + ["- `tests/`: automated checks", ""]
    return "\n".join(lines)


def _spec(title: str, goal: str, requirements: list[str], design: list[str]) -> str:
    out = [f"# Spec: {title}", "", "## Goal", goal, "", "## Requirements"]
    out += [f"- {r}" for r in requirements] + ["", "## Design"] + [f"- {d}" for d in design]
    out += ["", "## Verification", "- pytest unit tests for the core logic and for the command-line interface.",
            "- Run the tests in a virtualenv (`.venv`) before finishing.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------
# 2. Unit converter
# --------------------------------------------------------------------------------------------------

def converter(rng: random.Random) -> Task:
    pkg = _pick(rng, "convert", "unitconv", "measure", "unitcalc")
    who = _pick(rng, "a home baker", "a hiking group", "a science class", "a travel blogger")
    outline = _d(f"""
        Please build a command-line unit converter for {who}. Keep everything in this folder.

        What it should do
        - Convert lengths between kilometres, metres, centimetres, miles and feet (km, m, cm, mi, ft).
        - Convert weights between kilograms, grams, pounds and ounces (kg, g, lb, oz).
        - Convert temperatures between Celsius, Fahrenheit and Kelvin (c, f, k).
        - Show results rounded to two decimals.
        - Refuse conversions between different kinds of unit (like kilometres to kilograms) and unknown units,
          with a clear message instead of crashing.

        How people will use it
        - `python3 -m {pkg} 1 km m` prints `1000.00 m`.
        - `python3 -m {pkg} 32 f c` prints `0.00 c`.
        - `python3 -m {pkg} 1 kg lb` prints `2.20 lb`.

        Quality expectations
        - Organized and easy for the owner to extend with new units later, with a short guide.
        - Automated checks that prove the conversions are right, passing before you finish.
        """)
    units = _d('''
        """Unit tables: each linear unit maps to its size in the base unit of its kind."""
        from __future__ import annotations

        LENGTH = {"km": 1000.0, "m": 1.0, "cm": 0.01, "mi": 1609.344, "ft": 0.3048}
        WEIGHT = {"kg": 1000.0, "g": 1.0, "lb": 453.59237, "oz": 28.349523125}
        TEMPERATURE = ("c", "f", "k")


        def kind_of(unit: str) -> str:
            if unit in LENGTH:
                return "length"
            if unit in WEIGHT:
                return "weight"
            if unit in TEMPERATURE:
                return "temperature"
            raise ValueError(f"unknown unit: {unit}")
        ''')
    core = _d('''
        """Conversion logic, independent of the command line."""
        from __future__ import annotations

        from .units import LENGTH, WEIGHT, kind_of


        def _to_celsius(value: float, unit: str) -> float:
            return {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[unit]


        def _from_celsius(value: float, unit: str) -> float:
            return {"c": value, "f": value * 9 / 5 + 32, "k": value + 273.15}[unit]


        def convert(value: float, source: str, target: str) -> float:
            """Convert value from source to target unit; ValueError for unknown or mismatched units."""
            source, target = source.lower(), target.lower()
            kind = kind_of(source)
            if kind_of(target) != kind:
                raise ValueError(f"cannot convert {kind} ({source}) to {kind_of(target)} ({target})")
            if kind == "temperature":
                return _from_celsius(_to_celsius(value, source), target)
            table = LENGTH if kind == "length" else WEIGHT
            return value * table[source] / table[target]
        ''')
    cli = _d(f'''
        """Command-line interface for the converter."""
        from __future__ import annotations

        import argparse
        import sys

        from .core import convert


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(prog="{pkg}", description="Convert units.")
            parser.add_argument("value", type=float)
            parser.add_argument("source")
            parser.add_argument("target")
            args = parser.parse_args(argv)
            try:
                result = convert(args.value, args.source, args.target)
            except ValueError as exc:
                print(f"error: {{exc}}", file=sys.stderr)
                return 2
            print(f"{{result:.2f}} {{args.target.lower()}}")
            return 0
        ''')
    tests_core = _d(f'''
        import pytest

        from {pkg}.core import convert


        @pytest.mark.parametrize("value,src,dst,expected", [
            (1, "km", "m", 1000), (12, "ft", "m", 3.6576), (32, "f", "c", 0),
            (100, "c", "f", 212), (0, "c", "k", 273.15), (1, "kg", "lb", 2.2046), (16, "oz", "lb", 1)])
        def test_conversions(value, src, dst, expected):
            assert convert(value, src, dst) == pytest.approx(expected, rel=1e-3)


        @pytest.mark.parametrize("src,dst", [("km", "kg"), ("c", "m"), ("parsec", "m")])
        def test_rejects_bad_units(src, dst):
            with pytest.raises(ValueError):
                convert(1, src, dst)
        ''')
    tests_cli = _d(f'''
        from {pkg}.cli import main


        def test_output_format(capsys):
            assert main(["1", "km", "m"]) == 0
            assert capsys.readouterr().out.strip() == "1000.00 m"


        def test_error_is_reported(capsys):
            assert main(["1", "km", "kg"]) == 2
            assert "error" in capsys.readouterr().err
        ''')
    files = _common_files(pkg, "a command-line unit converter.")
    files.update({f"{pkg}/units.py": units, f"{pkg}/core.py": core, f"{pkg}/cli.py": cli,
                  "tests/test_core.py": tests_core, "tests/test_cli.py": tests_cli})
    spec = _spec(f"{pkg} unit converter", f"A command-line converter for {who}.",
                 ["Lengths: km, m, cm, mi, ft. Weights: kg, g, lb, oz. Temperatures: c, f, k.",
                  "Output `<value with two decimals> <target unit>`, e.g. `1000.00 m`.",
                  "Unknown units or different kinds (length vs weight) give an error message and exit code 2.",
                  "Units are case-insensitive."],
                 [f"`{pkg}/units.py`: unit tables (size of each unit in a base unit) and unit kinds.",
                  f"`{pkg}/core.py`: `convert()` for linear units via the base unit; temperatures via Celsius.",
                  f"`{pkg}/cli.py`: argparse front end; formatting and error output only.",
                  "Adding a unit means adding one table entry."])
    readme = _readme(pkg, "A command-line unit converter.", [f"python3 -m {pkg} 1 km m", f"python3 -m {pkg} 32 f c"],
                     {f"{pkg}/units.py": "unit tables (add new units here)", f"{pkg}/core.py": "conversion logic",
                      f"{pkg}/cli.py": "command-line interface"})
    checks = [Check(["1", "km", "m"], [r"1000\.00\s*m"]), Check(["32", "f", "c"], [r"0\.00\s*c"]),
              Check(["1", "kg", "lb"], [r"2\.20\s*lb"]), Check(["100", "c", "k"], [r"373\.15\s*k"]),
              Check(["3", "mi", "km"], [r"4\.83\s*km"]), Check(["1", "km", "kg"], [], ok=False),
              Check(["1", "parsec", "m"], [], ok=False)]
    return Task("converter", f"{pkg}-converter", pkg, outline,
                ["length", "weight", "temperature", "two decimals", "unknown", "test"], checks, files, spec, readme)


# --------------------------------------------------------------------------------------------------
# 3. Task tracker
# --------------------------------------------------------------------------------------------------

def tasks(rng: random.Random) -> Task:
    pkg = _pick(rng, "todo", "taskbox", "chores", "jobs")
    who = _pick(rng, "a busy parent", "a small design studio", "a volunteer group", "a home-renovation project")
    outline = _d(f"""
        I need a simple command-line to-do list for {who}. Put it all in this folder.

        What it should do
        - Add a task with a title and a priority (high, medium or low; medium if not given).
        - List tasks, highest priority first, each with a number people can refer to.
        - Mark a task as done; done tasks stay in the list but are clearly marked.
        - Remove a task.
        - Give a clear message for an unknown task number or priority instead of crashing.
        - Keep the list between runs.

        How people will use it
        - `python3 -m {pkg} add "Buy paint" --priority high` prints `added 1`.
        - `python3 -m {pkg} list` prints lines like `1 [ ] high Buy paint`, and `1 [x] high Buy paint` once done.
        - `python3 -m {pkg} done 1` prints `done 1`; `python3 -m {pkg} remove 1` prints `removed 1`.

        Quality expectations
        - Organized for later maintenance, with a short guide for the owner.
        - Automated checks that prove it works, passing before you finish.
        """)
    models = _d('''
        """Task records and list operations."""
        from __future__ import annotations

        from dataclasses import dataclass

        PRIORITIES = {"high": 0, "medium": 1, "low": 2}


        @dataclass
        class Task:
            id: int
            title: str
            priority: str = "medium"
            done: bool = False


        def add(tasks: list[Task], title: str, priority: str = "medium") -> Task:
            if priority not in PRIORITIES:
                raise ValueError(f"priority must be one of {', '.join(PRIORITIES)}")
            task = Task(max((t.id for t in tasks), default=0) + 1, title, priority)
            tasks.append(task)
            return task


        def find(tasks: list[Task], task_id: int) -> Task:
            for t in tasks:
                if t.id == task_id:
                    return t
            raise KeyError(f"no task {task_id}")


        def ordered(tasks: list[Task]) -> list[Task]:
            return sorted(tasks, key=lambda t: (PRIORITIES[t.priority], t.id))
        ''')
    storage = _d(f'''
        """Saving and loading the task list as JSON."""
        from __future__ import annotations

        import json
        import os
        from pathlib import Path

        from .models import Task


        def data_path() -> Path:
            return Path(os.environ.get("{pkg.upper()}_FILE", ".{pkg}.json"))


        def load() -> list[Task]:
            path = data_path()
            return [Task(**row) for row in json.loads(path.read_text())] if path.exists() else []


        def save(tasks: list[Task]) -> None:
            data_path().write_text(json.dumps([t.__dict__ for t in tasks], indent=2))
        ''')
    cli = _d(f'''
        """Command-line interface for the to-do list."""
        from __future__ import annotations

        import argparse
        import sys

        from . import storage
        from .models import add, find, ordered


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(prog="{pkg}", description="A small to-do list.")
            sub = parser.add_subparsers(dest="command", required=True)
            p_add = sub.add_parser("add")
            p_add.add_argument("title")
            p_add.add_argument("--priority", default="medium")
            sub.add_parser("list")
            for name in ("done", "remove"):
                sub.add_parser(name).add_argument("id", type=int)
            args = parser.parse_args(argv)
            tasks = storage.load()
            try:
                if args.command == "add":
                    print(f"added {{add(tasks, args.title, args.priority).id}}")
                elif args.command == "list":
                    for t in ordered(tasks):
                        print(f"{{t.id}} [{{'x' if t.done else ' '}}] {{t.priority}} {{t.title}}")
                elif args.command == "done":
                    find(tasks, args.id).done = True
                    print(f"done {{args.id}}")
                else:
                    tasks.remove(find(tasks, args.id))
                    print(f"removed {{args.id}}")
            except (KeyError, ValueError) as exc:
                print(f"error: {{exc.args[0]}}", file=sys.stderr)
                return 2
            storage.save(tasks)
            return 0
        ''')
    tests_models = _d(f'''
        import pytest

        from {pkg}.models import Task, add, find, ordered


        def test_add_assigns_increasing_ids_and_default_priority():
            tasks: list[Task] = []
            assert add(tasks, "a").id == 1 and add(tasks, "b", "high").id == 2
            assert tasks[0].priority == "medium"


        def test_ordered_by_priority_then_id():
            tasks: list[Task] = []
            add(tasks, "low one", "low"); add(tasks, "high one", "high"); add(tasks, "mid one")
            assert [t.title for t in ordered(tasks)] == ["high one", "mid one", "low one"]


        def test_errors():
            with pytest.raises(ValueError):
                add([], "x", "urgent")
            with pytest.raises(KeyError):
                find([], 3)
        ''')
    tests_cli = _d(f'''
        from {pkg}.cli import main


        def test_round_trip(tmp_path, monkeypatch, capsys):
            monkeypatch.setenv("{pkg.upper()}_FILE", str(tmp_path / "t.json"))
            main(["add", "Buy paint", "--priority", "high"]); main(["done", "1"])
            capsys.readouterr()
            main(["list"])
            assert capsys.readouterr().out.strip() == "1 [x] high Buy paint"
            assert main(["remove", "9"]) == 2
        ''')
    files = _common_files(pkg, "a command-line to-do list.")
    files.update({f"{pkg}/models.py": models, f"{pkg}/storage.py": storage, f"{pkg}/cli.py": cli,
                  "tests/test_models.py": tests_models, "tests/test_cli.py": tests_cli})
    spec = _spec(f"{pkg} to-do list", f"A command-line to-do list for {who}.",
                 ["Add a task with title and priority (high/medium/low, default medium); print `added <id>`.",
                  "List tasks highest priority first as `<id> [ ]|[x] <priority> <title>`.",
                  "`done <id>` marks done (kept in list); `remove <id>` deletes; print `done <id>` / `removed <id>`.",
                  "Unknown id or priority: error message, exit code 2, no crash.",
                  "Tasks persist between runs."],
                 [f"`{pkg}/models.py`: `Task` dataclass and list operations (pure functions).",
                  f"`{pkg}/storage.py`: JSON load/save, path from `{pkg.upper()}_FILE` or `.{pkg}.json`.",
                  f"`{pkg}/cli.py`: argparse commands add/list/done/remove; output only."])
    readme = _readme(pkg, "A small command-line to-do list.",
                     [f'python3 -m {pkg} add "Buy paint" --priority high', f"python3 -m {pkg} list",
                      f"python3 -m {pkg} done 1", f"python3 -m {pkg} remove 1"],
                     {f"{pkg}/models.py": "tasks and list operations", f"{pkg}/storage.py": "saving and loading",
                      f"{pkg}/cli.py": "the commands"})
    checks = [Check(["add", "Buy paint", "--priority", "high"], [r"added 1"]), Check(["add", "Sweep"], [r"added 2"]),
              Check(["add", "Call plumber", "--priority", "low"], [r"added 3"]),
              Check(["list"], [r"1 \[ \] high Buy paint[\s\S]*2 \[ \] medium Sweep[\s\S]*3 \[ \] low Call plumber"]),
              Check(["done", "2"], [r"done 2"]), Check(["list"], [r"2 \[x\] medium Sweep"]),
              Check(["remove", "1"], [r"removed 1"]), Check(["done", "9"], [], ok=False),
              Check(["add", "x", "--priority", "urgent"], [], ok=False)]
    return Task("tasks", f"{pkg}-list", pkg, outline, ["priority", "done", "remove", "list", "persist", "test"],
                checks, files, spec, readme)


# --------------------------------------------------------------------------------------------------
# 4. Text statistics
# --------------------------------------------------------------------------------------------------

def textstats(rng: random.Random) -> Task:
    pkg = _pick(rng, "textstats", "wordcount", "readstats", "prose")
    who = _pick(rng, "a writing club", "an English teacher", "a blogger", "a translator")
    outline = _d(f"""
        Make a command-line tool for {who} that reports statistics about a text file. Keep it in this folder.

        What it should do
        - Count words, lines and characters in a text file.
        - Show the five most common words (ignoring capitalisation and punctuation), most common first,
          with how often each appears; ties are broken alphabetically.
        - Estimate reading time at 200 words per minute, rounded up to whole minutes.
        - Give a clear message if the file does not exist instead of crashing.

        How people will use it
        - `python3 -m {pkg} essay.txt` prints:
          `words: 9`, `lines: 2`, `characters: 44`, `reading time: 1 min`, then `top words:` followed by
          lines like `the: 3`.

        Quality expectations
        - Organized so the owner can add new statistics later, with a short guide.
        - Automated checks that prove the counts are right, passing before you finish.
        """)
    analysis = _d('''
        """Text statistics, independent of files and the command line."""
        from __future__ import annotations

        import math
        import re
        from collections import Counter
        from dataclasses import dataclass

        WORD = re.compile(r"[a-z0-9']+")
        WORDS_PER_MINUTE = 200


        @dataclass
        class Stats:
            words: int
            lines: int
            characters: int
            reading_minutes: int
            top_words: list[tuple[str, int]]


        def words_of(text: str) -> list[str]:
            return [w.strip("'") for w in WORD.findall(text.lower()) if w.strip("'")]


        def analyse(text: str, top: int = 5) -> Stats:
            words = words_of(text)
            counts = sorted(Counter(words).items(), key=lambda kv: (-kv[1], kv[0]))
            return Stats(words=len(words), lines=len(text.splitlines()), characters=len(text),
                         reading_minutes=math.ceil(len(words) / WORDS_PER_MINUTE) if words else 0,
                         top_words=counts[:top])
        ''')
    report = _d('''
        """Formatting statistics for display."""
        from __future__ import annotations

        from .analysis import Stats


        def render(stats: Stats) -> str:
            lines = [f"words: {stats.words}", f"lines: {stats.lines}", f"characters: {stats.characters}",
                     f"reading time: {stats.reading_minutes} min", "top words:"]
            lines += [f"{word}: {count}" for word, count in stats.top_words]
            return "\\n".join(lines)
        ''')
    cli = _d(f'''
        """Command-line interface: read the file, analyse, print."""
        from __future__ import annotations

        import argparse
        import sys
        from pathlib import Path

        from .analysis import analyse
        from .report import render


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(prog="{pkg}", description="Text statistics for a file.")
            parser.add_argument("path")
            args = parser.parse_args(argv)
            try:
                text = Path(args.path).read_text(encoding="utf-8")
            except OSError as exc:
                print(f"error: cannot read {{args.path}}: {{exc.strerror}}", file=sys.stderr)
                return 2
            print(render(analyse(text)))
            return 0
        ''')
    tests_analysis = _d(f'''
        from {pkg}.analysis import analyse, words_of


        def test_counts_and_top_words():
            text = "The cat and the hat.\\nThe end, the END!\\n"
            stats = analyse(text)
            assert (stats.words, stats.lines, stats.characters) == (9, 2, len(text))
            assert stats.top_words[:2] == [("the", 4), ("end", 2)]


        def test_ties_alphabetical_and_reading_time():
            stats = analyse("b a c " * 150)
            assert stats.top_words == [("a", 150), ("b", 150), ("c", 150)]
            assert stats.reading_minutes == 3


        def test_punctuation_and_empty():
            assert words_of("Don't stop -- now!") == ["don't", "stop", "now"]
            assert analyse("").reading_minutes == 0
        ''')
    tests_cli = _d(f'''
        from {pkg}.cli import main


        def test_file_report(tmp_path, capsys):
            f = tmp_path / "a.txt"
            f.write_text("one two two\\n")
            assert main([str(f)]) == 0
            out = capsys.readouterr().out
            assert "words: 3" in out and "two: 2" in out


        def test_missing_file(capsys):
            assert main(["/nonexistent/file.txt"]) == 2
            assert "error" in capsys.readouterr().err
        ''')
    files = _common_files(pkg, "statistics for text files.")
    files.update({f"{pkg}/analysis.py": analysis, f"{pkg}/report.py": report, f"{pkg}/cli.py": cli,
                  "tests/test_analysis.py": tests_analysis, "tests/test_cli.py": tests_cli})
    spec = _spec(f"{pkg} text statistics", f"A command-line text statistics tool for {who}.",
                 ["Report words, lines and characters of a UTF-8 text file.",
                  "Top five words, case-insensitive, punctuation ignored, most common first, ties alphabetical.",
                  "Reading time: words / 200, rounded up to whole minutes.",
                  "Output lines `words: N`, `lines: N`, `characters: N`, `reading time: N min`, `top words:`, `word: N`.",
                  "Missing or unreadable file: error message, exit code 2, no crash."],
                 [f"`{pkg}/analysis.py`: tokenising and the `Stats` calculation (pure).",
                  f"`{pkg}/report.py`: formatting the statistics.",
                  f"`{pkg}/cli.py`: reads the file and prints the report.",
                  "New statistics: add a field in analysis and a line in report."])
    readme = _readme(pkg, "Statistics for text files.", [f"python3 -m {pkg} essay.txt"],
                     {f"{pkg}/analysis.py": "the calculations", f"{pkg}/report.py": "output formatting",
                      f"{pkg}/cli.py": "the command"})
    sample = "The cat and the hat.\nThe end, the END!\n"
    checks = [Check(["__SAMPLE__"], [r"words:\s*9", r"lines:\s*2", rf"characters:\s*{len(sample)}",
                                     r"reading time:\s*1 min", r"top words:", r"the:\s*4", r"end:\s*2"]),
              Check(["missing.txt"], [], ok=False)]
    task = Task("textstats", f"{pkg}-tool", pkg, outline,
                ["words", "lines", "characters", "reading time", "common", "test"], checks, files, spec, readme)
    task.sample = sample  # written to the check folder as sample.txt by the grader
    return task


# --------------------------------------------------------------------------------------------------
# 5. Inventory
# --------------------------------------------------------------------------------------------------

def inventory(rng: random.Random) -> Task:
    pkg = _pick(rng, "stock", "pantry", "inventory", "storeroom")
    who = _pick(rng, "a community food pantry", "a craft shop", "a school science lab", "a café")
    outline = _d(f"""
        Build a command-line stock keeper for {who}. Everything goes in this folder.

        What it should do
        - Add a quantity of an item (creating the item the first time).
        - Remove a quantity of an item, never letting stock go below zero.
        - Show the current stock of one item.
        - List items whose stock is below a threshold (default 5), lowest first.
        - Give clear messages for unknown items, non-positive quantities and over-removal instead of crashing.
        - Keep stock levels between runs.

        How people will use it
        - `python3 -m {pkg} add rice 10` prints `rice: 10`.
        - `python3 -m {pkg} remove rice 4` prints `rice: 6`.
        - `python3 -m {pkg} stock rice` prints `rice: 6`.
        - `python3 -m {pkg} low --threshold 8` prints one `item: quantity` line per low item.

        Quality expectations
        - Organized for maintenance, with a short guide for the owner.
        - Automated checks that prove it works, passing before you finish.
        """)
    core = _d('''
        """Stock levels and the rules for changing them."""
        from __future__ import annotations


        class StockError(ValueError):
            """A stock change that is not allowed."""


        def _check_quantity(quantity: int) -> None:
            if quantity <= 0:
                raise StockError("quantity must be a positive whole number")


        def add(stock: dict[str, int], item: str, quantity: int) -> int:
            _check_quantity(quantity)
            stock[item] = stock.get(item, 0) + quantity
            return stock[item]


        def remove(stock: dict[str, int], item: str, quantity: int) -> int:
            _check_quantity(quantity)
            if item not in stock:
                raise StockError(f"unknown item: {item}")
            if quantity > stock[item]:
                raise StockError(f"only {stock[item]} {item} in stock")
            stock[item] -= quantity
            return stock[item]


        def level(stock: dict[str, int], item: str) -> int:
            if item not in stock:
                raise StockError(f"unknown item: {item}")
            return stock[item]


        def low(stock: dict[str, int], threshold: int = 5) -> list[tuple[str, int]]:
            return sorted(((i, q) for i, q in stock.items() if q < threshold), key=lambda iq: (iq[1], iq[0]))
        ''')
    storage = _d(f'''
        """Saving and loading stock levels as JSON."""
        from __future__ import annotations

        import json
        import os
        from pathlib import Path


        def data_path() -> Path:
            return Path(os.environ.get("{pkg.upper()}_FILE", ".{pkg}.json"))


        def load() -> dict[str, int]:
            path = data_path()
            return json.loads(path.read_text()) if path.exists() else {{}}


        def save(stock: dict[str, int]) -> None:
            data_path().write_text(json.dumps(stock, indent=2, sort_keys=True))
        ''')
    cli = _d(f'''
        """Command-line interface for the stock keeper."""
        from __future__ import annotations

        import argparse
        import sys

        from . import storage
        from .core import StockError, add, level, low, remove


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(prog="{pkg}", description="Keep track of stock.")
            sub = parser.add_subparsers(dest="command", required=True)
            for name in ("add", "remove"):
                p = sub.add_parser(name)
                p.add_argument("item")
                p.add_argument("quantity", type=int)
            sub.add_parser("stock").add_argument("item")
            sub.add_parser("low").add_argument("--threshold", type=int, default=5)
            args = parser.parse_args(argv)
            stock = storage.load()
            try:
                if args.command == "add":
                    print(f"{{args.item}}: {{add(stock, args.item, args.quantity)}}")
                elif args.command == "remove":
                    print(f"{{args.item}}: {{remove(stock, args.item, args.quantity)}}")
                elif args.command == "stock":
                    print(f"{{args.item}}: {{level(stock, args.item)}}")
                else:
                    for item, quantity in low(stock, args.threshold):
                        print(f"{{item}}: {{quantity}}")
            except StockError as exc:
                print(f"error: {{exc}}", file=sys.stderr)
                return 2
            storage.save(stock)
            return 0
        ''')
    tests_core = _d(f'''
        import pytest

        from {pkg}.core import StockError, add, level, low, remove


        def test_add_remove_and_level():
            stock: dict[str, int] = {{}}
            assert add(stock, "rice", 10) == 10
            assert remove(stock, "rice", 4) == 6
            assert level(stock, "rice") == 6


        @pytest.mark.parametrize("action", [
            lambda s: remove(s, "rice", 99), lambda s: remove(s, "beans", 1),
            lambda s: add(s, "rice", 0), lambda s: level(s, "beans")])
        def test_rejected_changes(action):
            with pytest.raises(StockError):
                action({{"rice": 3}})


        def test_low_sorted_lowest_first():
            assert low({{"a": 4, "b": 1, "c": 9}}) == [("b", 1), ("a", 4)]
        ''')
    tests_cli = _d(f'''
        from {pkg}.cli import main


        def test_commands(tmp_path, monkeypatch, capsys):
            monkeypatch.setenv("{pkg.upper()}_FILE", str(tmp_path / "s.json"))
            main(["add", "rice", "10"]); main(["remove", "rice", "4"])
            capsys.readouterr()
            main(["stock", "rice"])
            assert capsys.readouterr().out.strip() == "rice: 6"
            assert main(["remove", "rice", "50"]) == 2
        ''')
    files = _common_files(pkg, "a command-line stock keeper.")
    files.update({f"{pkg}/core.py": core, f"{pkg}/storage.py": storage, f"{pkg}/cli.py": cli,
                  "tests/test_core.py": tests_core, "tests/test_cli.py": tests_cli})
    spec = _spec(f"{pkg} stock keeper", f"A command-line stock keeper for {who}.",
                 ["`add <item> <qty>` / `remove <item> <qty>` print the new level as `item: N`.",
                  "Stock never goes below zero; quantities must be positive whole numbers.",
                  "`stock <item>` prints `item: N`; unknown items are an error.",
                  "`low [--threshold N]` (default 5) lists items below the threshold, lowest first.",
                  "Errors: message, exit code 2, no crash. Stock persists between runs."],
                 [f"`{pkg}/core.py`: stock rules as functions over a dict; `StockError` for refused changes.",
                  f"`{pkg}/storage.py`: JSON load/save, path from `{pkg.upper()}_FILE` or `.{pkg}.json`.",
                  f"`{pkg}/cli.py`: argparse commands; output only."])
    readme = _readme(pkg, "A command-line stock keeper.",
                     [f"python3 -m {pkg} add rice 10", f"python3 -m {pkg} remove rice 4", f"python3 -m {pkg} stock rice",
                      f"python3 -m {pkg} low --threshold 8"],
                     {f"{pkg}/core.py": "stock rules", f"{pkg}/storage.py": "saving and loading",
                      f"{pkg}/cli.py": "the commands"})
    checks = [Check(["add", "rice", "10"], [r"rice:\s*10"]), Check(["remove", "rice", "4"], [r"rice:\s*6"]),
              Check(["add", "beans", "2"], [r"beans:\s*2"]), Check(["stock", "rice"], [r"rice:\s*6"]),
              Check(["low", "--threshold", "8"], [r"beans:\s*2[\s\S]*rice:\s*6"]),
              Check(["remove", "rice", "99"], [], ok=False), Check(["stock", "flour"], [], ok=False),
              Check(["add", "rice", "0"], [], ok=False)]
    return Task("inventory", f"{pkg}-keeper", pkg, outline,
                ["add", "remove", "stock", "threshold", "zero", "persist", "test"], checks, files, spec, readme)


# --------------------------------------------------------------------------------------------------
# 6. Grade book
# --------------------------------------------------------------------------------------------------

def gradebook(rng: random.Random) -> Task:
    pkg = _pick(rng, "grades", "gradebook", "marks", "classbook")
    who = _pick(rng, "a music teacher", "a tutoring centre", "a coding bootcamp", "a swimming coach")
    outline = _d(f"""
        Create a command-line grade book for {who}. Keep everything in this folder.

        What it should do
        - Add students.
        - Record a score from 0 to 100 for a student on a named assignment (re-recording replaces the old score).
        - Show a student's average and letter grade (A 90+, B 80+, C 70+, D 60+, otherwise F).
        - Show a report of every student, sorted by name.
        - Give clear messages for unknown students and scores outside 0-100 instead of crashing.
        - Keep the grade book between runs.

        How people will use it
        - `python3 -m {pkg} add-student Ana` prints `added Ana`.
        - `python3 -m {pkg} score Ana quiz1 92` prints `Ana quiz1: 92`.
        - `python3 -m {pkg} average Ana` prints `Ana: 87.50 B` (two decimals).
        - `python3 -m {pkg} report` prints one such line per student.

        Quality expectations
        - Organized for maintenance, with a short guide for the owner.
        - Automated checks that prove it works, passing before you finish.
        """)
    core = _d('''
        """Grade book rules: scores, averages and letter grades."""
        from __future__ import annotations

        LETTERS = [(90, "A"), (80, "B"), (70, "C"), (60, "D")]


        class GradeError(ValueError):
            """A grade-book change that is not allowed."""


        def add_student(book: dict[str, dict[str, float]], name: str) -> None:
            book.setdefault(name, {})


        def record(book: dict[str, dict[str, float]], name: str, assignment: str, score: float) -> None:
            if name not in book:
                raise GradeError(f"unknown student: {name}")
            if not 0 <= score <= 100:
                raise GradeError("score must be between 0 and 100")
            book[name][assignment] = score


        def average(book: dict[str, dict[str, float]], name: str) -> float:
            if name not in book:
                raise GradeError(f"unknown student: {name}")
            scores = list(book[name].values())
            return sum(scores) / len(scores) if scores else 0.0


        def letter(avg: float) -> str:
            return next((grade for cutoff, grade in LETTERS if avg >= cutoff), "F")
        ''')
    storage = _d(f'''
        """Saving and loading the grade book as JSON."""
        from __future__ import annotations

        import json
        import os
        from pathlib import Path


        def data_path() -> Path:
            return Path(os.environ.get("{pkg.upper()}_FILE", ".{pkg}.json"))


        def load() -> dict[str, dict[str, float]]:
            path = data_path()
            return json.loads(path.read_text()) if path.exists() else {{}}


        def save(book: dict[str, dict[str, float]]) -> None:
            data_path().write_text(json.dumps(book, indent=2, sort_keys=True))
        ''')
    cli = _d(f'''
        """Command-line interface for the grade book."""
        from __future__ import annotations

        import argparse
        import sys

        from . import storage
        from .core import GradeError, add_student, average, letter, record


        def line(book, name: str) -> str:
            avg = average(book, name)
            return f"{{name}}: {{avg:.2f}} {{letter(avg)}}"


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(prog="{pkg}", description="A small grade book.")
            sub = parser.add_subparsers(dest="command", required=True)
            sub.add_parser("add-student").add_argument("name")
            p = sub.add_parser("score")
            p.add_argument("name")
            p.add_argument("assignment")
            p.add_argument("score", type=float)
            sub.add_parser("average").add_argument("name")
            sub.add_parser("report")
            args = parser.parse_args(argv)
            book = storage.load()
            try:
                if args.command == "add-student":
                    add_student(book, args.name)
                    print(f"added {{args.name}}")
                elif args.command == "score":
                    record(book, args.name, args.assignment, args.score)
                    print(f"{{args.name}} {{args.assignment}}: {{args.score:g}}")
                elif args.command == "average":
                    print(line(book, args.name))
                else:
                    for name in sorted(book):
                        print(line(book, name))
            except GradeError as exc:
                print(f"error: {{exc}}", file=sys.stderr)
                return 2
            storage.save(book)
            return 0
        ''')
    tests_core = _d(f'''
        import pytest

        from {pkg}.core import GradeError, add_student, average, letter, record


        def test_average_and_replacement():
            book: dict = {{}}
            add_student(book, "Ana")
            record(book, "Ana", "quiz1", 92); record(book, "Ana", "quiz2", 70); record(book, "Ana", "quiz2", 83)
            assert average(book, "Ana") == 87.5


        @pytest.mark.parametrize("avg,grade", [(95, "A"), (90, "A"), (89.99, "B"), (70, "C"), (61, "D"), (12, "F")])
        def test_letters(avg, grade):
            assert letter(avg) == grade


        def test_errors():
            with pytest.raises(GradeError):
                record({{}}, "Bo", "x", 50)
            with pytest.raises(GradeError):
                record({{"Bo": {{}}}}, "Bo", "x", 101)
        ''')
    tests_cli = _d(f'''
        from {pkg}.cli import main


        def test_flow(tmp_path, monkeypatch, capsys):
            monkeypatch.setenv("{pkg.upper()}_FILE", str(tmp_path / "g.json"))
            main(["add-student", "Ana"]); main(["score", "Ana", "quiz1", "92"]); main(["score", "Ana", "quiz2", "83"])
            capsys.readouterr()
            main(["average", "Ana"])
            assert capsys.readouterr().out.strip() == "Ana: 87.50 B"
            assert main(["score", "Zed", "quiz1", "50"]) == 2
        ''')
    files = _common_files(pkg, "a command-line grade book.")
    files.update({f"{pkg}/core.py": core, f"{pkg}/storage.py": storage, f"{pkg}/cli.py": cli,
                  "tests/test_core.py": tests_core, "tests/test_cli.py": tests_cli})
    spec = _spec(f"{pkg} grade book", f"A command-line grade book for {who}.",
                 ["`add-student <name>` prints `added <name>`.",
                  "`score <name> <assignment> <0-100>` records (replacing an earlier score) and prints `name assignment: score`.",
                  "`average <name>` prints `name: avg(2 decimals) letter`; letters A 90+, B 80+, C 70+, D 60+, else F.",
                  "`report` prints one line per student sorted by name.",
                  "Unknown student or out-of-range score: message, exit code 2, no crash. Data persists."],
                 [f"`{pkg}/core.py`: grade-book rules over a dict; `GradeError` for refused changes.",
                  f"`{pkg}/storage.py`: JSON load/save, path from `{pkg.upper()}_FILE` or `.{pkg}.json`.",
                  f"`{pkg}/cli.py`: argparse commands; output only."])
    readme = _readme(pkg, "A command-line grade book.",
                     [f"python3 -m {pkg} add-student Ana", f"python3 -m {pkg} score Ana quiz1 92",
                      f"python3 -m {pkg} average Ana", f"python3 -m {pkg} report"],
                     {f"{pkg}/core.py": "grading rules", f"{pkg}/storage.py": "saving and loading",
                      f"{pkg}/cli.py": "the commands"})
    checks = [Check(["add-student", "Ana"], [r"added Ana"]), Check(["add-student", "Bo"], [r"added Bo"]),
              Check(["score", "Ana", "quiz1", "92"], [r"Ana quiz1:\s*92"]), Check(["score", "Ana", "quiz2", "83"], []),
              Check(["score", "Bo", "quiz1", "55"], []), Check(["average", "Ana"], [r"Ana:\s*87\.50\s*B"]),
              Check(["report"], [r"Ana:\s*87\.50\s*B[\s\S]*Bo:\s*55\.00\s*F"]),
              Check(["score", "Zed", "quiz1", "50"], [], ok=False), Check(["score", "Ana", "quiz3", "101"], [], ok=False)]
    return Task("gradebook", f"{pkg}-book", pkg, outline,
                ["student", "score", "average", "letter", "report", "persist", "test"], checks, files, spec, readme)


FAMILIES = [ledger, converter, tasks, textstats, inventory, gradebook]


def make_task(seed: int) -> Task:
    rng = random.Random(seed)
    return FAMILIES[seed % len(FAMILIES)](rng)
