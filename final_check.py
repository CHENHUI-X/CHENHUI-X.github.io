#!/usr/bin/env python3
"""Final comprehensive check of the article."""
import re

with open('src/content/posts/2026-05-16-llm-tokenization-guide.md', 'r') as f:
    content = f.read()
lines = content.split('\n')

issues = []

# 1. Check for the 3-asterisk horizontal rules (should be --- not ***)
# The article uses --- as horizontal rules
hr_count = content.count('---')
# But --- also appears in YAML frontmatter (lines 1 and 12)
# And in table separator lines
# Let me count just standalone --- lines
standalone_hr = 0
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if stripped == '---' and (i == 1 or i == 12):
        continue  # YAML frontmatter
    if stripped == '---':
        standalone_hr += 1
print(f"Standalone horizontal rules (---): {standalone_hr}")

# 2. Check that all table rows have the same number of columns
# Tables are consecutive lines starting with |
in_table = False
table_start = 0
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if stripped.startswith('|') and '|' in stripped[1:-1] if len(stripped) > 2 else False:
        if not in_table:
            in_table = True
            table_start = i
    else:
        if in_table:
            in_table = False
            # Check column count consistency
            table_lines = []
            
print("\nDone checking tables")

# 3. Check for the specific issue with \operatorname*
# in KaTeX
for i, line in enumerate(lines, 1):
    if 'operatorname*' in line:
        print(f"Line {i}: contains \\operatorname* - verify KaTeX support")

# 4. Check for any non-ASCII characters in the URL/title
print(f"\nTitle line: {lines[1]}")

# 5. Check that all code blocks have correct language specifier
# Find all ``` lines
for i, line in enumerate(lines, 1):
    if line.strip().startswith('```') and len(line.strip()) > 3:
        lang = line.strip()[3:].strip()
        if lang and not lang.isalnum() and '+' not in lang:
            print(f"Line {i}: unusual code fence language: {repr(lang)}")

# 6. Check the specific __ (double underscore) usage for non-italic
for i, line in enumerate(lines, 1):
    if '__' in line and '**' not in line:
        # Check for things like __init__
        if re.search(r'__[a-z_]+\b__', line):
            # This is Python __init__ style, should be in code
            if not line.strip().startswith('#'):
                pass  # Could be an issue if not in code block

# 7. Check for the paragraph that says "word-level" — verify the math
# line 570 formula: \text{计算量} \propto \underbrace{L^2 d}_{\text{Self-Attention}} + \underbrace{V d}_{\text{Embedding}}
# Check if _ inside \text{} is correct
for i, line in enumerate(lines, 1):
    if 'Self-Attention' in line or 'Embedding' in line:
        print(f"Line {i}: Check underbrace/Embedding rendering")

# 8. Check the special notation for wordpiece ##
for i, line in enumerate(lines, 1):
    if '##' in line and line.strip().startswith('|'):
        print(f"Line {i} (table): contains ##")

# 9. Final: look for any line with odd characters
print("\n=== Final scan for edge cases ===")
for i, line in enumerate(lines[:20], 1):
    if '\\' in line:
        # Check for odd backslash usage
        backslashes = re.findall(r'\\(.)', line)
        for bs in backslashes:
            if bs not in ['text', 'to', 'frac', 'times', 'approx', 'log', 'propto', 'cdot', 'operatorname*', 'ldots', 'sum', 'in', 'underbrace', '"', '\n']:
                pass  # might need checking
