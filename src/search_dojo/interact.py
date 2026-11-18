"""Steps, quizzes and progress shared by the walkthrough lessons."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.prompt import FloatPrompt, Prompt

from search_dojo.corpus import ROOT

PROGRESS_PATH = ROOT / ".dojo_progress.json"


@dataclass
class Session:
    """One run of a lesson.

    In a terminal it pauses between steps and asks quizzes. Anywhere else
    (pipes, tests, ``--no-pause``) it prints straight through and shows the
    quiz answers inline.
    """

    console: Console
    lesson: str
    total_steps: int
    interactive: bool
    correct: int = 0
    asked: int = 0

    @classmethod
    def start(cls, console: Console, lesson: str, total_steps: int, pause: bool) -> Session:
        interactive = pause and console.is_interactive and sys.stdin.isatty()
        return cls(console, lesson, total_steps, interactive)

    def step(self, number: int, title: str) -> None:
        """Print a step header, waiting for Enter first when interactive."""
        if number > 1 and self.interactive:
            self.console.input(
                f"\n[dim]Press Enter for step {number}/{self.total_steps}: {title}[/dim] "
            )
        self.console.print()
        self.console.rule(f"[bold]Step {number}/{self.total_steps} · {title}[/bold]")

    def quiz(self, question: str, options: list[str], answer: int, why: str) -> bool:
        """Ask a multiple-choice question. ``answer`` is the 0-based index of the right option."""
        self.console.print(f"\n[bold magenta]Quiz[/bold magenta]  {question}")
        for number, option in enumerate(options, start=1):
            self.console.print(f"  {number}. {option}")
        if not self.interactive:
            self.console.print(f"[green]Answer: {answer + 1}. {options[answer]}[/green]")
            self.console.print(why)
            return True
        self.asked += 1
        choices = [str(number) for number in range(1, len(options) + 1)]
        picked = int(Prompt.ask("Your answer", choices=choices, console=self.console)) - 1
        return self._grade(picked == answer, f"{answer + 1}. {options[answer]}", why)

    def guess(self, question: str, actual: float, tolerance: float, why: str) -> bool:
        """Ask for a number. Within ``tolerance`` of ``actual`` counts as right."""
        self.console.print(f"\n[bold magenta]Quiz[/bold magenta]  {question}")
        if not self.interactive:
            self.console.print(f"[green]Answer: {actual:.3f}[/green]")
            self.console.print(why)
            return True
        self.asked += 1
        value = FloatPrompt.ask("Your answer", console=self.console)
        return self._grade(abs(value - actual) <= tolerance, f"{actual:.3f}", why)

    def ask_text(self, prompt: str, default: str) -> str:
        """Free-text input when interactive, otherwise the default."""
        if not self.interactive:
            self.console.print(f"{prompt} [dim](using {default!r})[/dim]")
            return default
        return Prompt.ask(prompt, default=default, console=self.console)

    def ask_float(self, prompt: str, default: float) -> float:
        """Number input when interactive, otherwise the default."""
        if not self.interactive:
            self.console.print(f"{prompt} [dim](using {default})[/dim]")
            return default
        return FloatPrompt.ask(prompt, default=default, console=self.console)

    def finish(self) -> None:
        """Print the quiz score and record the lesson as done."""
        if not self.interactive:
            return
        if self.asked:
            self.console.print(
                f"\n[bold]Quiz score: {self.correct}/{self.asked}[/bold] for {self.lesson}."
            )
        save_progress(self.lesson, self.correct, self.asked)

    def _grade(self, ok: bool, answer: str, why: str) -> bool:
        if ok:
            self.correct += 1
            self.console.print("[bold green]Correct.[/bold green]")
        else:
            self.console.print(f"[bold red]Not quite.[/bold red] The answer is {answer}.")
        self.console.print(why)
        return ok


def load_progress(path: Path = PROGRESS_PATH) -> dict[str, dict[str, int]]:
    """Read finished lessons and their best quiz scores."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def save_progress(lesson: str, correct: int, asked: int, path: Path = PROGRESS_PATH) -> None:
    """Keep the best score per lesson."""
    data = load_progress(path)
    previous = data.get(lesson)
    if previous is None or correct >= previous.get("correct", 0):
        data[lesson] = {"correct": correct, "asked": asked}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def bar(value: float, width: int = 30) -> str:
    """A text bar for values in [0, 1]."""
    filled = max(0, min(width, round(value * width)))
    return "█" * filled + "[dim]" + "·" * (width - filled) + "[/dim]"
