#!/usr/bin/env python3
with open('src/content/posts/2026-05-16-llm-tokenization-guide.md', 'r') as f:
    content = f.read()

# Count ``` code blocks (triple backticks)
count = content.count('```')
print(f'Triple backtick count: {count}')
if count % 2 != 0:
    print('*** MISMATCHED CODE BLOCKS ***')
else:
    print(f'All code blocks properly closed ({count//2} blocks)')

# Check for unmatched backticks at line level
lines = content.split('\n')
in_code_block = False
for i, line in enumerate(lines, 1):
    if line.strip().startswith('```'):
        in_code_block = not in_code_block

# Count single backtick usage (outside triple backtick blocks)
lines = content.split('\n')
in_code_block = False
single_backtick_lines = []
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if stripped.startswith('```'):
        in_code_block = not in_code_block
        continue
    if not in_code_block and '`' in line and '``' not in line:
        cnt = line.count('`')
        if cnt % 2 != 0:
            single_backtick_lines.append((i, line.strip()[:60]))

if single_backtick_lines:
    print(f'\n*** LINES WITH UNMATCHED SINGLE BACKTICKS ***')
    for ln, l in single_backtick_lines:
        print(f'  Line {ln}: {l}')
else:
    print(f'\nNo unmatched single backtick issues found')
    
# Count all unmatched  
# Simple check: count all ` and see if even
all_backticks = content.count('`')
triple = content.count('```') * 3
# Remaining single backticks
single = all_backticks - triple
# Actually, each occurrence of ``` counts as 3 backticks
# But we've already counted them in `count`. 
# The total triple-backtick characters = count * 3
triple_chars = count * 3
remaining = all_backticks - triple_chars
print(f'\nTriple backtick chars: {triple_chars}')
print(f'All backtick chars: {all_backticks}')
print(f'Remaining single backticks: {remaining}')
if remaining % 2 != 0:
    print('*** UNMATCHED SINGLE BACKTICK DETECTED ***')
else:
    print('All single backticks are paired')
