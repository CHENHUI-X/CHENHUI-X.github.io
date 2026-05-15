#!/usr/bin/env python3
import re

with open('src/content/posts/2026-05-16-llm-tokenization-guide.md', 'rb') as f:
    raw = f.read()
content = raw.decode('utf-8')

# Check for trailing whitespace on EVERY line
lines = content.split('\n')
trailing = []
for i, line in enumerate(lines, 1):
    if line != line.rstrip():
        trailing.append((i, repr(line[-30:])))

if trailing:
    print(f"=== TRAILING WHITESPACE ({len(trailing)} lines) ===")
    for ln, rep in trailing:
        print(f"  Line {ln}: {rep}")
else:
    print("No trailing whitespace")

# Print all lines with any trailing whitespace
print("\n=== TRAILING WHITESPACE (full list) ===")
for ln, rep in trailing[:20]:
    print(f"  {ln}: end={rep}")

# Check quotes: " (ASCII 0x22) vs " (U+201C) vs " (U+201D)
ascii_quote = content.count('"')
left_curly = content.count('\u201c')
right_curly = content.count('\u201d')
left_single = content.count('\u2018')
right_single = content.count('\u2019')
print(f"\n=== QUOTE ANALYSIS ===")
print(f"ASCII double quotes (\") inside code blocks excluded")
print(f"Left curly double (U+201C): {left_curly}")
print(f"Right curly double (U+201D): {right_curly}")
print(f"Left curly single (U+2018): {left_single}")
print(f"Right curly single (U+2019): {right_single}")

# Check for ASCII double quotes outside code blocks
in_code = False
ascii_quotes_outside_code = []
for i, line in enumerate(lines, 1):
    if line.strip().startswith('```'):
        in_code = not in_code
        continue
    if not in_code:
        for j, c in enumerate(line):
            if c == '"':
                context = line[max(0,j-10):j+11]
                ascii_quotes_outside_code.append((i, j, repr(context)))

print(f"\nASCII double quotes outside code blocks: {len(ascii_quotes_outside_code)}")
for ln, col, ctx in ascii_quotes_outside_code[:20]:
    print(f"  Line {ln}, col {col}: {ctx}")

# Check for curly quotes inside code blocks
in_code = False
curly_quotes_in_code = []
for i, line in enumerate(lines, 1):
    if line.strip().startswith('```'):
        in_code = not in_code
        continue
    if in_code:
        for j, c in enumerate(line):
            if c in ('\u201c', '\u201d', '\u2018', '\u2019'):
                context = line[max(0,j-10):j+11]
                curly_quotes_in_code.append((i, j, repr(context), repr(c)))

print(f"\nCurly quotes inside code blocks: {len(curly_quotes_in_code)}")
for ln, col, ctx, c in curly_quotes_in_code[:10]:
    print(f"  Line {ln}, col {col}: char={c}, context={ctx}")
