import re
import glob

for filepath in sorted(glob.glob('src/content/posts/*.md')):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    issues = []
    in_code = False
    for i in range(len(lines) - 1):
        line = lines[i].rstrip()
        next_line = lines[i+1].rstrip()

        if line.strip().startswith('```'):
            in_code = not in_code
        if in_code:
            continue

        # Text line (non-empty, non-heading) followed by $$ (no blank line between)
        if next_line.strip() == '$$' and line.strip() and not line.strip().startswith('#'):
            prev_empty = i == 0 or not lines[i-1].strip()
            if not prev_empty:
                issues.append((i+1, 'text->$$', line.strip()[:60]))

        # $$ followed by text line (no blank line between)
        if line.strip() == '$$' and next_line.strip() and not next_line.strip().startswith('#'):
            next_next_empty = i+2 >= len(lines) or not lines[i+2].strip()
            if not next_next_empty and next_line.strip() != '$$':
                issues.append((i+2, '$$->text', next_line.strip()[:60]))

    if issues:
        print(f'\n=== {filepath.split("/")[-1]} ===')
        for line_no, typ, content in issues:
            print(f'  L{line_no} [{typ}]: {content}')
