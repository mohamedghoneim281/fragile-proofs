"""Lossless SMT-LIB 2 helpers.

We never *interpret* a script.  We only
  * split it into top-level commands (keeping every comment/space so that
    ``render(split(text)) == text`` always holds),
  * consistently rename the symbols that the script itself declares, and
  * permute runs of consecutive ``assert`` commands.

Both rewrites keep the meaning of the query (alpha-renaming of declared
symbols; commutativity of conjunction), which is what makes the variants
"harmless input changes".
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Set

# comment | string | |quoted symbol| | whitespace | paren | atom | anything else
TOKEN_RE = re.compile(r';[^\n]*|"(?:[^"]|"")*"|\|[^|]*\||\s+|[()]|[^\s()";|]+|.', re.S)

DECL_COMMANDS = frozenset(
    {
        "declare-fun",
        "declare-const",
        "define-fun",
        "define-fun-rec",
        "define-const",
        "declare-sort",
        "define-sort",
    }
)

# Never rename these, even if a script (weirdly) declares them.
RESERVED = frozenset(
    {
        "Int", "Bool", "Real", "String", "Array", "BitVec", "true", "false",
        "and", "or", "not", "=>", "=", "distinct", "ite", "xor", "forall",
        "exists", "let", "match", "par", "_", "!", "as", "select", "store",
        "assert", "check-sat", "declare-fun", "declare-const", "define-fun",
    }
)


class Chunk:
    """A top-level command (``is_cmd``) or the trivia (spaces/comments) between commands."""

    __slots__ = ("text", "name", "is_cmd")

    def __init__(self, text: str, name: Optional[str], is_cmd: bool):
        self.text = text
        self.name = name
        self.is_cmd = is_cmd


def canon(tok: str) -> str:
    """Canonical symbol name: ``|abc|`` and ``abc`` are the same symbol in SMT-LIB."""
    if len(tok) >= 2 and tok[0] == "|" and tok[-1] == "|":
        return tok[1:-1]
    return tok


def _is_atom(tok: str) -> bool:
    c = tok[0]
    return not (c in '();"' or tok.isspace())


def split_script(text: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    depth = 0
    start = 0
    trivia_start = 0
    name: Optional[str] = None
    expect_name = False
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        c = tok[0]
        if depth == 0:
            if c == "(":
                if m.start() > trivia_start:
                    chunks.append(Chunk(text[trivia_start:m.start()], None, False))
                depth = 1
                start = m.start()
                name = None
                expect_name = True
            continue
        if c == "(":
            depth += 1
            expect_name = False
        elif c == ")":
            depth -= 1
            expect_name = False
            if depth == 0:
                chunks.append(Chunk(text[start:m.end()], name, True))
                trivia_start = m.end()
        elif c == ";" or tok.isspace():
            continue
        elif expect_name:
            name = canon(tok) if c != '"' else None
            expect_name = False
    if depth != 0:
        raise ValueError("unbalanced parentheses in SMT-LIB script")
    if trivia_start < len(text):
        chunks.append(Chunk(text[trivia_start:], None, False))
    return chunks


def render(chunks: List[Chunk]) -> str:
    return "".join(c.text for c in chunks)


def declared_name(cmd_text: str) -> Optional[str]:
    """Name declared by a ``(declare-fun NAME ...)``-style command, or None."""
    stage = 0  # 0: want "(", 1: want command name, 2: want declared symbol
    for m in TOKEN_RE.finditer(cmd_text[:5000]):
        tok = m.group(0)
        if tok.isspace() or tok[0] == ";":
            continue
        if stage == 0:
            if tok != "(":
                return None
            stage = 1
        elif stage == 1:
            if not _is_atom(tok):
                return None
            stage = 2
        else:
            return canon(tok) if _is_atom(tok) else None
    return None


def rename_symbols(text: str, mapping: Dict[str, str]) -> str:
    """Replace every occurrence of the mapped symbols (never inside strings/comments)."""

    def repl(m: "re.Match[str]") -> str:
        tok = m.group(0)
        if not _is_atom(tok):
            return tok
        new = mapping.get(canon(tok))
        return tok if new is None else new

    return TOKEN_RE.sub(repl, text)


class Script:
    """Parsed view of one query file (parsing happens once, variants are cheap)."""

    def __init__(self, text: str):
        self.text = text
        self.chunks = split_script(text)
        self._declared: Optional[List[str]] = None
        self._atoms: Optional[Set[str]] = None

    # ---- facts about the script -------------------------------------------------
    def count(self, command: str) -> int:
        return sum(1 for c in self.chunks if c.is_cmd and c.name == command)

    @property
    def declared(self) -> List[str]:
        if self._declared is None:
            seen: Set[str] = set()
            out: List[str] = []
            for ch in self.chunks:
                if ch.is_cmd and ch.name in DECL_COMMANDS:
                    n = declared_name(ch.text)
                    if n and n not in seen and n not in RESERVED:
                        seen.add(n)
                        out.append(n)
            self._declared = out
        return self._declared

    @property
    def atoms(self) -> Set[str]:
        if self._atoms is None:
            self._atoms = {
                canon(m.group(0)) for m in TOKEN_RE.finditer(self.text) if _is_atom(m.group(0))
            }
        return self._atoms

    def assert_runs(self) -> List[List[int]]:
        """Chunk indices of maximal runs (length >= 2) of consecutive ``assert`` commands.

        Only trivia may sit between asserts of a run, so a permutation can never move
        an assert across a declaration, push/pop or check-sat.
        """
        runs: List[List[int]] = []
        cur: List[int] = []
        for i, ch in enumerate(self.chunks):
            if not ch.is_cmd:
                continue
            if ch.name == "assert":
                cur.append(i)
            else:
                if len(cur) > 1:
                    runs.append(cur)
                cur = []
        if len(cur) > 1:
            runs.append(cur)
        return runs

    @property
    def can_rename(self) -> bool:
        return len(self.declared) > 0

    @property
    def can_shuffle(self) -> bool:
        return len(self.assert_runs()) > 0

    # ---- rewrites -----------------------------------------------------------------
    def rename_mapping(self, rng: random.Random) -> Dict[str, str]:
        used = set(self.atoms)
        mapping: Dict[str, str] = {}
        for name in self.declared:
            while True:
                new = "q%010x" % rng.getrandbits(40)
                if new not in used:
                    used.add(new)
                    break
            mapping[name] = new
        return mapping

    def shuffled_text(self, rng: random.Random) -> str:
        texts = [ch.text for ch in self.chunks]
        for run in self.assert_runs():
            items = [texts[i] for i in run]
            rng.shuffle(items)
            for i, t in zip(run, items):
                texts[i] = t
        return "".join(texts)

    def renamed_text(self, rng: random.Random, base_text: Optional[str] = None) -> str:
        return rename_symbols(self.text if base_text is None else base_text, self.rename_mapping(rng))
