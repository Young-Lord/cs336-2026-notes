#!/usr/bin/env python3
"""Convert half-width punctuation to full-width in Chinese prose of markdown notes.

Safety rules:
- Never touch fenced code blocks, inline code spans, LaTeX math ($..$ / $$..$$ / \\(..\\)),
  YAML frontmatter, or table separator rows.
- Only convert a punctuation char when it is adjacent to a CJK character
  (e.g. "你好,世界" -> "你好，世界"), leaving English/Latin text intact.
"""

import re
import sys
from pathlib import Path

CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef\U00020000-\U0002ebef"
CJK_RE = re.compile(f"[{CJK}]")

# characters that may legally follow a sentence-ending period
END_FOLLOWERS = set(" \t\n\r。，；：？！）》」』”\"'…-")

PLACEHOLDER_START = "\x00"
masked: dict[str, str] = {}


def mask(text: str) -> str:
    key = f"{PLACEHOLDER_START}{len(masked)}{PLACEHOLDER_START}"
    masked[key] = text
    return key


def unmask(text: str) -> str:
    return re.sub(
        f"{PLACEHOLDER_START}\\d+{PLACEHOLDER_START}",
        lambda m: masked[m.group(0)],
        text,
    )


def mask_spans(line: str) -> str:
    """Mask inline code, inline math and \\(..\\) spans so their content is untouched."""
    # inline code `...`
    line = re.sub(r"`[^`]*`", lambda m: mask(m.group(0)), line)
    # \( ... \)
    line = re.sub(r"\\\(.*?\\\)", lambda m: mask(m.group(0)), line)
    # $...$ (not $), avoid touching $ blocks
    line = re.sub(r"(?<!\$)\$(?!\$)([^$\n]*?)\$(?!\$)", lambda m: mask(m.group(0)), line)
    return line


def convert_parentheses(line: str) -> str:
    """Convert matched () pairs to full-width when the pair sits in a Chinese context.

    A pair is "in Chinese context" if the character right before '(' or right
    after ')' is a CJK character. Unmatched parens fall back to CJK adjacency.
    """
    stack = []
    pairs: set[int] = set()
    for i, ch in enumerate(line):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                open_i = stack.pop()
                before_open = line[open_i - 1] if open_i > 0 else ""
                after_close = line[i + 1] if i + 1 < len(line) else ""
                if CJK_RE.match(before_open) or CJK_RE.match(after_close):
                    pairs.add(open_i)
                    pairs.add(i)
            else:
                # unmatched close: convert if preceded by CJK
                before = line[i - 1] if i > 0 else ""
                if CJK_RE.match(before):
                    pairs.add(i)
    chars = list(line)
    for i in pairs:
        if chars[i] == "(":
            chars[i] = "（"
        elif chars[i] == ")":
            chars[i] = "）"
    # remaining unmatched '(' opens: convert if followed by CJK
    for i, ch in enumerate(line):
        if ch == "(" and i not in pairs:
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if CJK_RE.match(nxt):
                chars[i] = "（"
    return "".join(chars)


def convert_prose(line: str) -> str:
    line = mask_spans(line)
    line = convert_parentheses(line)
    out = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        prev = line[i - 1] if i > 0 else ""
        nxt = line[i + 1] if i + 1 < n else ""

        prev_cjk = bool(CJK_RE.match(prev))
        next_cjk = bool(CJK_RE.match(nxt))

        if ch in ",;:?!" and (prev_cjk or next_cjk):
            out.append({" ": " ", ",": "，", ";": "；", ":": "：", "?": "？", "!": "！"}[ch])
        elif ch == ".":
            if prev_cjk and nxt == ".":
                # start of a dot-run; handle run below
                j = i
                while j < n and line[j] == ".":
                    j += 1
                run = j - i
                if run == 3 and prev_cjk:
                    out.append("……")
                elif run == 2 and prev_cjk:
                    out.append("…")
                else:
                    out.append("." * run)
                i = j
                continue
            elif prev_cjk and (nxt == "" or nxt in END_FOLLOWERS):
                out.append("。")
            elif prev_cjk and next_cjk:
                out.append("。")
            else:
                out.append(".")
        else:
            out.append(ch)
        i += 1
    return unmask("".join(out))


def process_file(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    in_frontmatter = False
    in_code = False
    in_block_math = False
    changed = 0
    for line in lines:
        stripped = line.strip()
        if not in_code and not in_frontmatter and stripped.startswith("```"):
            in_code = True
            out.append(line)
            continue
        if in_code:
            if stripped.startswith("```"):
                in_code = False
            out.append(line)
            continue
        if not in_frontmatter and stripped == "---":
            # frontmatter start
            in_frontmatter = True
            out.append(line)
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            out.append(line)
            continue
        # block math $$ ... $$ possibly spanning lines
        if "$$" in line:
            if not in_block_math and stripped.startswith("$$"):
                in_block_math = True
                out.append(line)
                continue
            if in_block_math:
                if stripped.endswith("$$"):
                    in_block_math = False
                out.append(line)
                continue
        if in_block_math:
            out.append(line)
            continue
        # table separator row like | :--- | ---: |
        if re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", stripped):
            out.append(line)
            continue
        new_line = convert_prose(line)
        if new_line != line:
            changed += 1
        out.append(new_line)
    text = "".join(out)
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    targets = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            targets.extend(sorted(p.rglob("*.md")))
        else:
            targets.append(p)
    total = 0
    for f in targets:
        c = process_file(f)
        if c:
            print(f"{f}: {c} lines changed")
        total += c
    print(f"TOTAL lines changed: {total}")


if __name__ == "__main__":
    sys.exit(main())
