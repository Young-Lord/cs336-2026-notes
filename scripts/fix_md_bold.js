#!/usr/bin/env node
/* Fix markdown emphasis that fails to parse when `**` is adjacent to
 * punctuation (CommonMark flanking rules treat （）？！。"" as punctuation).
 *
 * Safety principle: regex rules only fire on unambiguous broken shapes
 * (e.g. closing ** after full-width paren followed by a letter). The two
 * ambiguous shapes are handled by parity-aware scanning:
 *   - rule 5: an OPENER ** directly before （, preceded by a letter
 *             (a legit closing ** is always at odd parity)
 *   - rule 6: a CLOSER ** preceded by full-width punctuation and followed
 *             by a letter (a legit opening ** is always at even parity)
 */
const fs = require("fs");
const path = require("path");

const QUOTE = '["“”]';
const LETTER = /[\u4e00-\u9fffA-Za-z]/;
const PUNCT = /[。？！；：、…]/;
const PLACEHOLDER_START = "\x00";
const masked = {};

function mask(text) {
  const key = `${PLACEHOLDER_START}${Object.keys(masked).length}${PLACEHOLDER_START}`;
  masked[key] = text;
  return key;
}

function unmask(text) {
  return text.replace(new RegExp(`${PLACEHOLDER_START}\\d+${PLACEHOLDER_START}`, "g"), (m) => masked[m]);
}

function maskSpans(line) {
  line = line.replace(/`[^`]*`/g, (m) => mask(m));
  line = line.replace(/\\\(.*?\\\)/g, (m) => mask(m));
  line = line.replace(/(?<!\$)\$(?!\$)([^$\n]*?)\$(?!\$)/g, (m) => mask(m));
  return line;
}

// positions of "**" pairs in the string
function starPositions(line) {
  const pos = [];
  for (let i = 0; i + 1 < line.length; i++) {
    if (line[i] === "*" && line[i + 1] === "*") { pos.push(i); i++; }
  }
  return pos;
}

// X**（A）B**  ->  X（A）**B**   (broken OPENER directly before a full-width paren)
function fixOpenerBeforeParen(line) {
  const positions = starPositions(line);
  const edits = [];
  const full = /（([^（）\n]*?)）([^*\n]*?)\*\*/;
  const fullOpenHalfClose = /（([^（）\n]*?)\)([^*\n]*?)\*\*/;
  const half = /\(([^()\n]*?)\)([^*\n]*?)\*\*/;
  const halfOpenFullClose = /\(([^()\n]*?）)([^*\n]*?)\*\*/;
  for (let k = 0; k < positions.length; k++) {
    const p = positions[k];
    if (k % 2 !== 0) continue;               // must be an opener (even parity)
    if (!LETTER.test(line[p - 1] || "")) continue; // broken only when preceded by a letter
    const open = line[p + 2];
    const re = open === "（" ? full : half;
    if (open !== "（" && open !== "(") continue;
    const m = re.exec(line.slice(p + 2)) || (open === "（" ? fullOpenHalfClose : halfOpenFullClose).exec(line.slice(p + 2));
    if (m) edits.push([p, p + 2 + m[0].length, `（${m[1]}）**${m[2]}**`]);
  }
  // apply right-to-left so indices stay valid
  for (let i = edits.length - 1; i >= 0; i--) {
    const [s, e, rep] = edits[i];
    line = line.slice(0, s) + rep + line.slice(e);
  }
  return line;
}

// same, but paren pair is full-open / half-close: （A)
function fixOpenerBeforeParenMixed(line) {
  const positions = starPositions(line);
  const edits = [];
  for (let k = 0; k < positions.length; k++) {
    const p = positions[k];
    if (k % 2 !== 0) continue;
    if (!LETTER.test(line[p - 1] || "")) continue;
    if (line[p + 2] !== "（") continue;
    const m = /（([^（）\n]*?)\)([^*\n]*?)\*\*/.exec(line.slice(p + 2));
    if (m) edits.push([p, p + 2 + m[0].length, `（${m[1]}）**${m[2]}**`]);
  }
  for (let i = edits.length - 1; i >= 0; i--) {
    const [s, e, rep] = edits[i];
    line = line.slice(0, s) + rep + line.slice(e);
  }
  return line;
}

// **X？**Z  ->  **X**？Z   (broken CLOSER ** preceded by full-width punct, followed by letter)
function fixCloserAfterPunct(line) {
  const positions = starPositions(line);
  const edits = [];
  for (let k = 0; k < positions.length; k++) {
    const p = positions[k];
    if (k % 2 === 0) continue;               // must be a closer (odd parity)
    if (!PUNCT.test(line[p - 1] || "")) continue;  // punct immediately before
    if (!LETTER.test(line[p + 2] || "")) continue; // letter immediately after
    const openerPos = positions[k - 1];
    const content = line.slice(openerPos + 2, p - 1);
    if (content.includes("*")) continue;     // must be a clean bold span
    edits.push([openerPos, p + 2, `**${content}**${line[p - 1]}`]);
  }
  for (let i = edits.length - 1; i >= 0; i--) {
    const [s, e, rep] = edits[i];
    line = line.slice(0, s) + rep + line.slice(e);
  }
  return line;
}

function fixLine(line) {
  line = maskSpans(line);
  const CJK_LETTER = "\\u4e00-\\u9fffA-Za-z";
  // 1a. X**"A"（B）**  ->  X"**A**"（B）
  line = line.replace(
    new RegExp(`(?<=[${CJK_LETTER}])\\*\\*(${QUOTE})([^*\\n]+?)(${QUOTE})（([^（）\\n]+?)）\\*\\*`, "g"),
    '$1**$2**$3（$4）'
  );
  // 1b. X**"A"(B)**  ->  X"**A**"（B）
  line = line.replace(
    new RegExp(`(?<=[${CJK_LETTER}])\\*\\*(${QUOTE})([^*\\n]+?)(${QUOTE})\\(([^()\\n]+?)\\)\\*\\*`, "g"),
    '$1**$2**$3（$4）'
  );
  // 2. X**"A"**  ->  X"**A**"
  line = line.replace(
    new RegExp(`(?<=[${CJK_LETTER}])\\*\\*(${QUOTE})([^*\\n]+?)(${QUOTE})\\*\\*`, "g"),
    '$1**$2**$3'
  );
  // 3. **X（Y）**Z  ->  **X**（Y）Z  (closing ** after full-width paren + letter)
  line = line.replace(/(?<!\*)\*\*([^*\n]+?)（([^（）\n]+?)）\*\*(?=[\u4e00-\u9fffA-Za-z])/g, '**$1**（$2）');
  // 4. **X(Y)**Z  ->  **X**(Y)Z
  line = line.replace(/(?<!\*)\*\*([^*\n]+?)\(([^()\n]+?)\)\*\*(?=[\u4e00-\u9fffA-Za-z])/g, '**$1**($2)');
  // 5. X**（A）B**  ->  X（A）**B**
  line = fixOpenerBeforeParen(line);
  line = fixOpenerBeforeParenMixed(line);
  // 6. **X？**Z  ->  **X**？Z
  line = fixCloserAfterPunct(line);
  // 7. **X（Y)**Z  ->  **X**（Y）Z  (full-width open, half-width close)
  line = line.replace(/(?<!\*)\*\*([^*\n]+?)（([^（）\n]+?)\)\*\*(?=[\u4e00-\u9fffA-Za-z])/g, '**$1**（$2）');
  return unmask(line);
}

function processFile(file) {
  const lines = fs.readFileSync(file, "utf-8").split("\n");
  const out = [];
  let inFence = false;
  let inFrontmatter = false;
  let inBlockMath = false;
  let changed = 0;
  for (let line of lines) {
    const stripped = line.trim();
    if (!inFence && !inFrontmatter && stripped.startsWith("```")) { inFence = true; out.push(line); continue; }
    if (inFence) { if (stripped.startsWith("```")) inFence = false; out.push(line); continue; }
    if (!inFrontmatter && stripped === "---") { inFrontmatter = true; out.push(line); continue; }
    if (inFrontmatter) { if (stripped === "---") inFrontmatter = false; out.push(line); continue; }
    if (stripped.startsWith("$$")) {
      if (stripped === "$$") inBlockMath = !inBlockMath;
      else if (!stripped.endsWith("$$")) inBlockMath = true;
      out.push(line); continue;
    }
    if (inBlockMath) { if (stripped.endsWith("$$")) inBlockMath = false; out.push(line); continue; }
    // iterate until stable (max 5 passes) to resolve cascading fixes
    let current = line;
    for (let pass = 0; pass < 5; pass++) {
      const next = fixLine(current);
      if (next === current) break;
      current = next;
    }
    if (current !== line) changed++;
    out.push(current);
  }
  if (changed) fs.writeFileSync(file, out.join("\n"), "utf-8");
  return changed;
}

let total = 0;
const files = [];
(function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const f = path.join(dir, e.name);
    if (e.isDirectory()) walk(f);
    else if (e.name.endsWith(".md")) files.push(f);
  }
})(process.argv[2] || "docs/lectures");

for (const f of files) {
  const c = processFile(f);
  if (c) console.log(`${path.relative(process.cwd(), f)}: ${c} lines changed`);
  total += c;
}
console.log(`TOTAL lines changed: ${total}`);
