#!/usr/bin/env python3
import re

with open('src/content/posts/2026-05-16-llm-tokenization-guide.md', 'r') as f:
    content = f.read()

# Check all formulas for potential KaTeX issues
# Find all math blocks (both display and inline)
# First, let's extract display math blocks
# We need to handle both multi-line $$...$$ and single-line $$...$$
display_blocks = []

lines = content.split('\n')
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    if stripped.startswith('$$'):
        block_lines = [line]
        i += 1
        # Check if it's a single-line $$...$$
        if stripped.endswith('$$') and len(stripped) > 4:
            # Single-line block: $$content$$
            pass
        else:
            # Multi-line block
            while i < len(lines):
                block_lines.append(lines[i])
                if lines[i].strip() == '$$':
                    break
                i += 1
        block_text = '\n'.join(block_lines)
        display_blocks.append(block_text)
    i += 1

print(f"=== Display Math Blocks ({len(display_blocks)}) ===")
for idx, block in enumerate(display_blocks):
    # Extract just the LaTeX content
    if block.strip().startswith('$$') and block.strip().endswith('$$') and len(block.strip()) > 4:
        content_latex = block.strip()[2:-2].strip()
    else:
        lines_b = block.split('\n')
        content_latex = '\n'.join(lines_b[1:-1]).strip()
    print(f"\nBlock {idx+1}:")
    print(f"  Content: {content_latex[:80]}...")
    
    # Check for common KaTeX pitfalls
    
    # 1. Check for unsupported commands
    unsupported = []
    for cmd in ['\operatorname*']:
        if cmd in content_latex:
            print(f"  NOTE: Contains {cmd} - check KaTeX support")
    
    # 2. Check for unescaped & inside align environments
    if '&' in content_latex and 'align' not in content_latex:
        pass  # & inside text is fine
    
    # 3. Check balanced braces
    open_b = content_latex.count('{')
    close_b = content_latex.count('}')
    if open_b != close_b:
        print(f"  *** UNBALANCED BRACES: {open_b} open, {close_b} close")

# Now check inline math $...$
print(f"\n\n=== Inline Math Check ===")
# Simpler approach: extract all math content from the raw file
raw_without_display = re.sub(r'\$\$[\s\S]*?\$\$', '', content)
# Find $...$ patterns (but NOT $$)
inline_matches = re.findall(r'\$([^$\n]+?)\$', raw_without_display)
print(f"Found {len(inline_matches)} inline math expressions")
for m in inline_matches[:20]:
    print(f"  ${m}$")
    open_b = m.count('{')
    close_b = m.count('}')
    if open_b != close_b:
        print(f"    *** UNBALANCED BRACES: {open_b} open, {close_b} close")

print("\n\n=== Checking for other potential issues ===")

# Check for Unicode dash/hyphen issues
# The article uses both — (em dash) and - (hyphen)
em_dash_count = content.count('—')
hyphen_count = content.count(' - ')
print(f"Em dashes (—): {em_dash_count}")
print(f"Hyphens with spaces ( - ): {hyphen_count}")

# Check for trailing whitespace
for i, line in enumerate(lines, 1):
    if line != line.rstrip():
        print(f"Line {i}: trailing whitespace detected: {repr(line[-20:])}")
        break
else:
    print("No trailing whitespace issues found (first pass)")

# Check for non-ASCII issues in code blocks
in_code = False
for i, line in enumerate(lines, 1):
    if line.strip().startswith('```'):
        in_code = not in_code
        continue
    if in_code:
        # Check for non-ASCII characters in code
        non_ascii = [c for c in line if ord(c) > 127]
        if non_ascii:
            print(f"Line {i} (code block): non-ASCII char {repr(non_ascii[0])}: {line.strip()[:60]}")
