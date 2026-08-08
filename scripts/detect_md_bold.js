#!/usr/bin/env node
/* Detect literal `*` in markdown prose (emphasis that failed to parse due to
 * full-width punctuation adjacent to `**` markers, per CommonMark flanking rules).
 * Fenced code, block math ($$), inline code and inline math are skipped. */
const fs = require("fs");
const path = require("path");
const MarkdownIt = require("markdown-it");
const md = new MarkdownIt();

const root = path.resolve(process.argv[2] || "docs/lectures");

function walkFiles(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkFiles(full, out);
    else if (entry.name.endsWith(".md")) out.push(full);
  }
  return out;
}

// Replace code/math spans with non-`*` placeholders so they never read as emphasis
function maskSpans(line) {
  const masked = {};
  let n = 0;
  const mask = (m) => (masked[`\x00${n++}\x00`] = m, `\x00${n - 1}\x00`);
  line = line.replace(/`[^`]*`/g, mask);
  line = line.replace(/\\\(.*?\\\)/g, mask);
  line = line.replace(/(?<!\$)\$(?!\$)([^$\n]*?)\$(?!\$)/g, mask);
  return line;
}

let total = 0;
for (const file of walkFiles(root)) {
  const lines = fs.readFileSync(file, "utf-8").split("\n");
  let inFence = false;
  let inFrontmatter = false;
  let inBlockMath = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const stripped = line.trim();
    if (!inFence && !inFrontmatter && stripped.startsWith("```")) { inFence = true; continue; }
    if (inFence) { if (stripped.startsWith("```")) inFence = false; continue; }
    if (!inFrontmatter && stripped === "---") { inFrontmatter = true; continue; }
    if (inFrontmatter) { if (stripped === "---") inFrontmatter = false; continue; }
    if (stripped.startsWith("$$")) {
      if (stripped === "$$") inBlockMath = !inBlockMath;
      else if (!stripped.endsWith("$$")) inBlockMath = true;
      continue;
    }
    if (inBlockMath) { if (stripped.endsWith("$$")) inBlockMath = false; continue; }
    if (!line.includes("*")) continue;
    const tokens = md.parse(maskSpans(line), {});
    const bad = [];
    (function walk(ts) {
      for (const t of ts) {
        if (t.type === "text" && t.content.includes("*")) bad.push(t.content);
        if (t.children) walk(t.children);
      }
    })(tokens);
    if (bad.length) {
      total++;
      console.log(`${path.relative(process.cwd(), file)}:${i + 1}: ${line.trim().slice(0, 120)}`);
      console.log(`    literal: ${JSON.stringify(bad.slice(0, 3).map((b) => b.trim().slice(0, 60)))}`);
    }
  }
}
console.log(`\nTOTAL broken lines: ${total}`);
