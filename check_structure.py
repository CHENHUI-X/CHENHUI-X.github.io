#!/usr/bin/env python3
import re

with open('src/content/posts/2026-05-16-llm-tokenization-guide.md', 'r') as f:
    content = f.read()

# Remove code blocks before checking
stripped = re.sub(r'```[\s\S]*?```', '', content)

# Check for **bold** and __italic__ markers
bold_starts = stripped.count('**')
print(f'Bold markers (**): {bold_starts}')
if bold_starts % 2 != 0:
    print('*** UNMATCHED BOLD MARKERS ***')

# Check for unmatched parentheses
lines = stripped.split('\n')
for i, line in enumerate(lines, 1):
    open_paren = line.count('(')
    close_paren = line.count(')')
    if open_paren != close_paren:
        print(f'Line {i}: MISMATCHED PARENS: {open_paren} open, {close_paren} close')
        print(f'  Content: {line[:80]}')

# Check for unmatched brackets
for i, line in enumerate(lines, 1):
    open_b = line.count('[')
    close_b = line.count(']')
    if open_b != close_b:
        print(f'Line {i}: MISMATCHED BRACKETS: {open_b} open, {close_b} close')
        print(f'  Content: {line[:80]}')

print("\nAll structural checks complete")
print(f"Total lines: {len(lines)}")

# Count total non-code lines
non_code_lines = [l for l in lines if l.strip() and not l.strip().startswith('```') and not l.strip().startswith('|')]
print(f"Non-code, non-table significant lines: {len(non_code_lines)}")
