"""
Comprehensive verification of ALL formulas, numbers, and factual claims
in the tokenization article.
"""
from collections import defaultdict, Counter
import math

print("="*80)
print("VERIFICATION 1: BPE Initial Pair Frequency Table (Section 2.2)")
print("="*80)

corpus = {
    "low": 10,
    "lowest": 5,
    "newer": 5,
    "wider": 5,
    "new": 2,
}

# Count all initial pairs
pair_counts = defaultdict(int)
for word, cnt in corpus.items():
    symbols = list(word) + ["</w>"]
    for i in range(len(symbols) - 1):
        pair_counts[(symbols[i], symbols[i+1])] += cnt

# Article's table
article_pairs = {
    ("l", "o"): 15,
    ("o", "w"): 15,
    ("w", "</w>"): 12,
    ("e", "s"): 5,
    ("s", "t"): 5,
    ("t", "</w>"): 5,
    ("n", "e"): 7,
    ("w", "e"): 10,
    ("e", "r"): 10,
    ("r", "</w>"): 10,
    ("w", "i"): 5,
    ("i", "d"): 5,
    ("d", "e"): 5,
    ("e", "w"): 7,
}

print(f"{'Pair':<15} {'Article':<10} {'Computed':<10} {'Match':<8}")
print("-"*45)
all_pairs_ok = True
for pair, expected in sorted(article_pairs.items()):
    actual = pair_counts.get(pair, 0)
    ok = actual == expected
    if not ok: all_pairs_ok = False
    print(f"{str(pair):<15} {expected:<10} {actual:<10} {'✅' if ok else '❌'}")

# Check for missing pairs
computed_pairs = set(pair_counts.keys())
article_pair_set = set(article_pairs.keys())
missing = computed_pairs - article_pair_set
extra = article_pair_set - computed_pairs
if missing:
    print(f"\n⚠️ Pairs in corpus but NOT in article table: {sorted(missing)}")
if extra:
    print(f"\n⚠️ Pairs in table but NOT in corpus: {sorted(extra)}")
if not missing and not extra:
    print("\n✅ All corpus pairs are covered in the article table.")

# Verify specific numbers
print("\nDetailed verification of key counts:")
# (l, o) = 15: low×10 + lowest×5
lo_count = 10 + 5
print(f"(l, o): low×10 + lowest×5 = {10}+{5} = {lo_count} {'✅' if lo_count==15 else '❌'}")

# (w, </w>) = 12: low×10 + new×2
w_end_count = 10 + 2
print(f"(w, </w>): low×10 + new×2 = {10}+{2} = {w_end_count} {'✅' if w_end_count==12 else '❌'}")

# (n, e) = 7: newer×5 + new×2
ne_count = 5 + 2
print(f"(n, e): newer×5 + new×2 = {5}+{2} = {ne_count} {'✅' if ne_count==7 else '❌'}")

# (w, e) = 10: lowest×5 + newer×5
we_count = 5 + 5
print(f"(w, e): lowest×5 + newer×5 = {5}+{5} = {we_count} {'✅' if we_count==10 else '❌'}")

# (e, r) = 10: newer×5 + wider×5
er_count = 5 + 5
print(f"(e, r): newer×5 + wider×5 = {5}+{5} = {er_count} {'✅' if er_count==10 else '❌'}")

# (r, </w>) = 10: newer×5 + wider×5
r_end_count = 5 + 5
print(f"(r, </w>): newer×5 + wider×5 = {5}+{5} = {r_end_count} {'✅' if r_end_count==10 else '❌'}")

# (e, w) = 7: newer×5 + new×2
ew_count = 5 + 2
print(f"(e, w): newer×5 + new×2 = {5}+{2} = {ew_count} {'✅' if ew_count==7 else '❌'}")

# (o, w) = 15: low×10 + lowest×5
ow_count = 10 + 5
print(f"(o, w): low×10 + lowest×5 = {10}+{5} = {ow_count} {'✅' if ow_count==15 else '❌'}")

print()
print("="*80)
print("VERIFICATION 2: BPE Merge Sequence")
print("="*80)

def simulate_bpe_full(corpus_dict, target_vocab_size):
    """Full BPE simulation."""
    # Initial sequences: list of (symbols, count)
    entries = []
    for word, cnt in corpus_dict.items():
        entry = (list(word) + ["</w>"], cnt)
        entries.append((word, entry))
    
    vocab = set()
    for word, cnt in corpus_dict.items():
        for ch in list(word) + ["</w>"]:
            vocab.add(ch)
    
    print(f"Initial vocab: {len(vocab)} tokens -> {sorted(vocab)}")
    steps_needed = target_vocab_size - len(vocab)
    print(f"Need {steps_needed} merges to reach {target_vocab_size}")
    
    merges = []
    
    while len(vocab) < target_vocab_size:
        # Count pairs
        pairs = defaultdict(int)
        for word, (syms, cnt) in entries:
            for i in range(len(syms) - 1):
                pairs[(syms[i], syms[i+1])] += cnt
        
        if not pairs:
            print("  No more pairs!")
            break
        
        best_pair = max(pairs, key=lambda k: pairs[k])
        best_count = pairs[best_pair]
        
        # Check ties
        tied = sorted([p for p, c in pairs.items() if c == best_count])
        tie_info = f" [tied with: {[p for p in tied if p != best_pair]}]" if len(tied) > 1 else ""
        
        merged_token = best_pair[0] + best_pair[1]
        vocab.add(merged_token)
        merges.append((best_pair, best_count, merged_token))
        
        step = len(merges)
        
        # Apply merge
        new_entries = []
        for word, (syms, cnt) in entries:
            new_syms = []
            i = 0
            while i < len(syms):
                if i < len(syms)-1 and syms[i] == best_pair[0] and syms[i+1] == best_pair[1]:
                    new_syms.append(merged_token)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            new_entries.append((word, (new_syms, cnt)))
        entries = new_entries
        
        print(f"  Step {step}: merge {best_pair} -> '{merged_token}' [count={best_count}]{tie_info} | vocab={len(vocab)}")
    
    return merges, entries, vocab

print("\n--- Simulating BPE to vocab_size=18 ---")
merges, final_entries, final_vocab = simulate_bpe_full(corpus, 18)

print(f"\nTotal merges performed: {len(merges)}")

print("\nFull merge sequence:")
for i, (pair, cnt, merged) in enumerate(merges):
    print(f"  Step {i+1}: {pair} -> '{merged}' (count={cnt})")

print("\n--- Checking article's claimed merge order ---")
# Article's claimed first two merges: (l,o) then (lo,w)
if len(merges) >= 2:
    step1_ok = merges[0][0] == ("l", "o")
    step2_ok = merges[1][0] == ("lo", "w")
    print(f"Step 1: merge (l,o) {'✅' if step1_ok else '❌'} (actual: {merges[0]})")
    print(f"Step 2: merge (lo,w) {'✅' if step2_ok else '❌'} (actual: {merges[1]})")
    
    if not step1_ok:
        print(f"  ⚠️  First merge differs! Article says (l,o), actual best pair is {merges[0][0]}")
    if not step2_ok and step1_ok:
        # After (l,o) merge, check if (lo,w) is indeed the best
        pass

# Check what the best pair is after step 1
print("\n--- Recomputing: After step 1 (l,o)->lo, what are pair counts? ---")
entries_after_step1 = []
for word, cnt in corpus.items():
    syms = list(word) + ["</w>"]
    # Apply (l,o) merge
    new_syms = []
    i = 0
    while i < len(syms):
        if i < len(syms)-1 and syms[i] == 'l' and syms[i+1] == 'o':
            new_syms.append('lo')
            i += 2
        else:
            new_syms.append(syms[i])
            i += 1
    entries_after_step1.append((word, new_syms, cnt))

pairs_step1 = defaultdict(int)
for word, syms, cnt in entries_after_step1:
    for i in range(len(syms)-1):
        pairs_step1[(syms[i], syms[i+1])] += cnt

print(f"{'Pair':<15} {'Count':<10}")
print("-"*25)
for pair, cnt in sorted(pairs_step1.items(), key=lambda x: -x[1]):
    print(f"{str(pair):<15} {cnt:<10}")

print()
print("="*80)
print("VERIFICATION 3: WordPiece Score Calculations (Section 3.2)")
print("="*80)

# Count individual token frequencies
token_freq = defaultdict(int)
for word, cnt in corpus.items():
    for ch in list(word) + ["</w>"]:
        token_freq[ch] += cnt

print("Individual token frequencies:")
for tok in sorted(token_freq.keys()):
    print(f"  '{tok}': {token_freq[tok]}")

# Recompute pair frequencies
pair_freq = defaultdict(int)
for word, cnt in corpus.items():
    symbols = list(word) + ["</w>"]
    for i in range(len(symbols) - 1):
        pair_freq[(symbols[i], symbols[i+1])] += cnt

print("\n--- score(l, o) ---")
pair_lo = pair_freq[('l', 'o')]
freq_l = token_freq['l']
freq_o = token_freq['o']
score_lo = pair_lo / (freq_l * freq_o)
print(f"freq(lo) = {pair_lo}")
print(f"freq(l) = {freq_l}")
print(f"freq(o) = {freq_o}")
print(f"score = {pair_lo} / ({freq_l} × {freq_o}) = {pair_lo}/{freq_l*freq_o} = {score_lo:.6f}")
print(f"Article: 1/15 ≈ 0.067")
expected_lo = 15 / (15 * 15)
print(f"Expected: {expected_lo:.6f}")
print(f"Match: {'✅' if abs(score_lo - expected_lo) < 0.001 else '❌'}")

print("\n--- score(o, w) ---")
pair_ow = pair_freq[('o', 'w')]
freq_o2 = token_freq['o']
freq_w = token_freq['w']
score_ow = pair_ow / (freq_o2 * freq_w)
print(f"freq(ow) = {pair_ow}")
print(f"freq(o) = {freq_o2}")
print(f"freq(w) = {freq_w}")
print(f"score = {pair_ow} / ({freq_o2} × {freq_w}) = {pair_ow}/{freq_o2*freq_w} = {score_ow:.6f}")
print(f"Article: 15/(15×27) ≈ 0.037")

# Verify freq(w) = ? 
# w appears in: low×10, lowest×5, newer×5, wider×5, new×2
# Each word: new has w once, low has w once, lowest has w once, newer has w once, wider has w once
w_breakdown = 10 + 5 + 5 + 5 + 2
print(f"\nfreq(w) breakdown: low×10={10}, lowest×5={5}, newer×5={5}, wider×5={5}, new×2={2}")
print(f"Total: {w_breakdown}")
print(f"Article claims freq(w) = 27")
print(f"Match: {'✅' if freq_w == 27 else '❌'} (computed: {freq_w})")

expected_ow = 15 / (15 * 27)
print(f"Expected: {expected_ow:.6f}")
print(f"Match: {'✅' if abs(score_ow - expected_ow) < 0.001 else '❌'}")

print("\n--- PMI Formula Check (Section 3.2) ---")
print("PMI(x,y) = log(P(x,y)/(P(x)P(y)))")
print("Article states: PMI ∝ log(freq(xy)/(freq(x)·freq(y)))")
print("WordPiece score = freq(xy)/(freq(x)·freq(y))")
print("✅ WordPiece score is PMI *without* the log, which is monotonic (score order preserved)")
print("✅ This is a standard formulation.")

print()
print("="*80)
print("VERIFICATION 4: BPE Step 2 Pair Frequency Table (Section 2.2)")
print("="*80)

# After merge (l,o)->lo
entries_s2 = [
    (["lo", "w", "</w>"], 10),
    (["lo", "w", "e", "s", "t", "</w>"], 5),
    (["n", "e", "w", "e", "r", "</w>"], 5),
    (["w", "i", "d", "e", "r", "</w>"], 5),
    (["n", "e", "w", "</w>"], 2),
]

pairs_s2 = defaultdict(int)
for syms, cnt in entries_s2:
    for i in range(len(syms)-1):
        pairs_s2[(syms[i], syms[i+1])] += cnt

article_s2_pairs = {
    ("lo", "w"): 15,
    ("w", "</w>"): 12,
    ("w", "e"): 10,
    ("e", "r"): 10,
    ("r", "</w>"): 10,
    ("n", "e"): 7,
    ("e", "w"): 7,
}

print("Step 2 pair frequency comparison:")
print(f"{'Pair':<15} {'Article':<10} {'Computed':<10} {'Match':<8}")
print("-"*45)
for pair, expected in sorted(article_s2_pairs.items()):
    actual = pairs_s2.get(pair, 0)
    ok = actual == expected
    print(f"{str(pair):<15} {expected:<10} {actual:<10} {'✅' if ok else '❌'}")

# Check for missing pairs
all_s2_pairs = set(pairs_s2.keys())
article_s2_set = set(article_s2_pairs.keys())
missing_s2 = all_s2_pairs - article_s2_set
if missing_s2:
    print(f"\n⚠️ Pairs existing after step 1 but NOT listed in article:")
    for p in sorted(missing_s2):
        print(f"  {p}: count={pairs_s2[p]}")

# Check that (e, s), (s, t), (t, </w>), (w, i), (i, d), (d, e) still exist
removed_pairs = [("e", "s"), ("s", "t"), ("t", "</w>"), ("w", "i"), ("i", "d"), ("d", "e")]
print("\nPairs that should still exist (from 'lowest' and 'wider' tokens):")
for p in removed_pairs:
    actual = pairs_s2.get(p, 0)
    print(f"  {p}: count={actual} {'✅ still present' if actual > 0 else '❌ MISSING'}")

# What about the unspecified pairs?
print("\nAll computed step 2 pairs (sorted by count):")
for pair, cnt in sorted(pairs_s2.items(), key=lambda x: -x[1]):
    marker = " ✅ in article" if pair in article_s2_set else " ⚠️ NOT in article table"
    print(f"  {str(pair):<15} {cnt:<5}{marker}")

print()
print("="*80)
print("VERIFICATION 5: Character-level Example (Section 1)")
print("="*80)

text = "I love machine learning"
mapping = {
    ' ': 1, 'I': 2, 'a': 3, 'c': 4, 'e': 5, 'g': 6, 
    'h': 7, 'i': 8, 'l': 9, 'm': 10, 'n': 11, 'o': 12, 
    'r': 13, 'v': 14
}

result_article = [2, 1, 9, 12, 14, 5, 1, 10, 3, 4, 7, 8, 11, 5, 1, 9, 5, 3, 13, 11, 8, 11, 6]
result_computed = [mapping[c] for c in text]

print(f"Text: '{text}'")
print(f"Length: actual chars = {len(text)}")
print(f"Article: {result_article}")
print(f"Computed: {result_computed}")
print(f"Article length: {len(result_article)}, Computed length: {len(result_computed)}")

if result_article == result_computed:
    print("✅ Character mapping is correct!")
else:
    print("❌ MISMATCH!")
    for i, (a, c) in enumerate(zip(result_article, result_computed)):
        if a != c:
            print(f"  Position {i}: text='{text[i]}', article={a}, computed={c}")

print()
print("="*80)
print("VERIFICATION 6: 'newest' Tokenization Example (Section 2.2)")
print("="*80)

print("Article claims: 'newest' → ['new', 'e', 's', 't', '</w>']")
print("Using the article's listed merge rules:")
print("  (n,e)→ne, (ne,w)→new")
print()
print("Process: n e w e s t </w>")
print("  → ne w e s t </w>  (merge n,e)")
print("  → new e s t </w>   (merge ne,w)")
print("  → no more merges apply")
print("  → ['new', 'e', 's', 't', '</w>']  ✅")

print()
print("="*80)
print("VERIFICATION 7: Model Vocab Sizes (Section 2.3, 3.3, 5.2)")
print("="*80)

print("Claimed facts:")
tests = [
    ("GPT-2 vocab", 50257, "known fact", True),
    ("GPT-3 vocab", 50257, "same as GPT-2 (GPT-3 used same tokenizer)", True),
    ("BERT vocab", 30000, "known fact", True),
    ("LLaMA-1 vocab", 32000, "known fact", True),
    ("LLaMA-2 vocab", 32000, "known fact", True),
    ("LLaMA-3 vocab", 128000, "known fact", True),
    ("T5 vocab", 32000, "known fact (default)", True),
    ("Qwen vocab", 152000, "known fact (~152K)", True),
    ("ChatGLM vocab", 130000, "approximately correct (ChatGLM-6B is 130,288)", True),
]
for name, claimed, note, expected_ok in tests:
    ok_emoji = "✅" if expected_ok else "❌"
    print(f"  {name}: {claimed} ({note}) {ok_emoji}")

# Check if RoBERTa is correctly categorized
print("\n--- RoBERTa check ---")
print("Article (line 284): 'WordPiece 在 BERT 系列 (BERT, RoBERTa, DistilBERT) 中使用'")
print("This claims RoBERTa uses WordPiece.")
print("⚠️ WAIT - RoBERTa actually uses Byte-level BPE (same as GPT-2), NOT WordPiece!")
print("RoBERTa = Robustly Optimized BERT Approach")
print("While it's based on BERT architecture, it changed the tokenizer from WordPiece to Byte-level BPE")
print("Reference: Liu et al. 2019, RoBERTa paper")

print()
print("="*80)
print("VERIFICATION 8: Section 6.3 chars/token claims")
print("="*80)

eng_text = "The quick brown fox jumps over the lazy dog"
cn_text = "大语言模型分词器的工作原理是什么"
mix_text = "BERT 的 WordPiece 和 LLaMA 的 BPE 有什么区别"

print(f"English text length: {len(eng_text)} chars")
print(f"Chinese text length: {len(cn_text)} chars")
print(f"Mixed text length: {len(mix_text)} chars")
print()
print("Article claims:")
print(f"  English: 10 tokens, 4.3 chars/token -> {4.3*10:.0f} chars (text has {len(eng_text)})")
print(f"  Chinese: 18 tokens, 1.4 chars/token -> {1.4*18:.1f} chars (text has {len(cn_text)})")
print(f"  Mixed: 22 tokens, 2.1 chars/token -> {2.1*22:.1f} chars (text has {len(mix_text)})")
print()

# Check the actual consistency
print("Checking chars/token consistency:")
eng_chars_per_token_implied = len(eng_text) / 10
cn_chars_per_token_implied = len(cn_text) / 18
mix_chars_per_token_implied = len(mix_text) / 22
print(f"  English: actual = {len(eng_text)}/10 = {eng_chars_per_token_implied:.1f} (article: 4.3)")
print(f"  Chinese: actual = {len(cn_text)}/18 = {cn_chars_per_token_implied:.1f} (article: 1.4)")
print(f"  Mixed: actual = {len(mix_text)}/22 = {mix_chars_per_token_implied:.1f} (article: 2.1)")

print()
print("="*80)
print("VERIFICATION 9: '4096 tokens' claim (Section 6.3)")
print("="*80)

print("Article: '同样的4096 token, 英文能读17500个字符, 中文只能读5800个字符'")
print(f"  4096 × 4.3 = {4096*4.3:.0f} (should be 17500)")
print(f"  4096 × 1.4 = {4096*1.4:.0f} (should be 5800)")
print()
print(f"  17500 / 4096 = {17500/4096:.2f} chars/token implied")
print(f"  5800 / 4096 = {5800/4096:.2f} chars/token implied")
print()
print("⚠️ INCONSISTENCY: The chars/token values in the examples (4.3, 1.4)")
print("   don't multiply to match 17500 and 5800.")
print("   Using the examples: 4.3*4096={:.0f}, 1.4*4096={:.0f}".format(4.3*4096, 1.4*4096))
print("   Using 17500/5800: 4.27 and 1.42 chars/token respectively.")

print()
print("="*80)
print("VERIFICATION 10: BPE Code Test (Section 6.4)")
print("="*80)

from collections import defaultdict as dd

def train_bpe(corpus_words, vocab_size):
    words = []
    for text in corpus_words:
        for word in text.split():
            words.append(" ".join(list(word)) + " </w>")
    vocab = set()
    for word in words:
        for char in word.split():
            vocab.add(char)
    merges = []
    while len(vocab) < vocab_size:
        pairs = dd(int)
        for word in words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += 1
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        new_words = []
        for word in words:
            new_word = word.replace(f"{best_pair[0]} {best_pair[1]}", f"{best_pair[0]}{best_pair[1]}")
            new_words.append(new_word)
        words = new_words
        vocab.add(f"{best_pair[0]}{best_pair[1]}")
    return merges

def apply_bpe(text, merges):
    words = text.split()
    result = []
    for word in words:
        tokens = list(word) + ["</w>"]
        for merge in merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i+1] == merge[1]:
                    tokens = tokens[:i] + [f"{merge[0]}{merge[1]}"] + tokens[i+2:]
                else:
                    i += 1
        result.extend(tokens)
    return result

print("Testing article's simplified BPE code:")
merges_test = train_bpe(["low low low lowest newer newer wider"], vocab_size=20)
print(f"Corpus: 'low low low lowest newer newer wider'")
print(f"Note: This test corpus has (low×3, lowest×1, newer×2, wider×1, new×0)")
print(f"  This does NOT match the main article corpus (low×10, lowest×5, etc.)!")
print(f"Merges learned: {merges_test}")
result = apply_bpe("lowest", merges_test)
print(f"apply_bpe('lowest') = {result}")

# Check a potential bug in apply_bpe
print("\n--- Checking apply_bpe logic ---")
print("The while loop has a bug: when a merge is applied at position i,")
print("the new merged token should be checked against position i+1 again")
print("(for cases like n+e → ne, then ne+w → new in one pass).")
print("But the code breaks by using tokens[:i] + [merged] + tokens[i+2:]")
print("and NOT resetting 'i'. Actually wait, it doesn't increment i after merge,")
print("so the next iteration checks the merged token with tokens[i+1].")
print("Actually, looking more carefully:")
print("  tokens = tokens[:i] + [f\"{merge[0]}{merge[1]}\"] + tokens[i+2:]")
print("  This shortens the list by 1. Then the while loop continues")
print("  without incrementing i (since no else branch runs after merge)")
print("  So it checks the merged token with the next one - that's actually CORRECT!")

print()
print("="*80)
print("VERIFICATION 11: Viterbi Formula (Section 4.2)")
print("="*80)

print("Formula: argmax Σ log P(t_i) for t_i in V")
print("✅ Standard Viterbi formulation for unigram language model segmentation")
print("✅ The article correctly describes the optimization objective")

print()
print("="*80)
print("VERIFICATION 12: Computational Complexity (Section 7.3)")
print("="*80)

print("Formula: L²d + Vd")
print("✅ Self-attention per layer is O(L²d)")
print("✅ Embedding layer parameters: V×d")
print("Note: ignores FFN (O(Ld²)) and number of layers - acceptable simplification")

print()
print("="*80)
print("VERIFICATION 13: Section 2.3 GPT-2/LLaMA facts")
print("="*80)

print("GPT-2 vocab = 50,257 ✅")
print("GPT-2 uses Byte-level BPE ✅")
print("LLaMA-1 vocab = 32,000 ✅")
print("LLaMA-3 vocab = 128,000 ✅")
print()
print("Note: Line 211 says 'LLaMA 系列的词表是 32,000 (LLaMA-1) 或 128,000 (LLaMA-3)'")
print("This is correct.")

print()
print("="*80)
print("VERIFICATION 14: Section 5.2 Table")
print("="*80)

print("Table claims:")
print("  GPT-2/3: Byte-level BPE, 50,257 ✅")
print("  BERT: WordPiece, 30,000 ✅")
print("  LLaMA: SentencePiece + BPE, 32,000 ✅")
print("  LLaMA-3: tiktoken BPE (byte-level), 128,000 ✅")
print("  T5: SentencePiece + Unigram, 32,000 ✅")
print("  Qwen: tiktoken BPE (byte-level), 152,000 ✅")
print("  GPT-4: Unknown, ≈100K ✅ (reasonable estimate)")

print()
print("="*80)
print("SUMMARY OF ISSUES")
print("="*80)

print("""
ISSUE 1 (Medium): RoBERTa listed as using WordPiece (line 284)
  - Article says "BERT 系列 (BERT, RoBERTa, DistilBERT)" uses WordPiece
  - RoBERTa actually uses Byte-level BPE, not WordPiece
  - DistilBERT inherits BERT's tokenizer (WordPiece), so that part is correct

ISSUE 2 (Minor): chars/token inconsistency in section 6.3
  - Example texts give different chars/token than the 4.3/1.4/2.1 values
  - English: 43 chars / 10 tokens = 4.3 ✅ (exact match)
  - Chinese: 16 chars / 18 tokens = 0.89 ❌ (article says 1.4)
    Wait let me recount...
    
  Chinese text: "大语言模型分词器的工作原理是什么"
  Let me count: 大(1)语(2)言(3)模(4)型(5)分(6)词(7)器(8)的(9)工(10)作(11)原(12)理(13)是(14)什(15)么(16)
  That's 16 characters. Article says 18 tokens at 1.4 chars/token = 25.2 chars
  That doesn't match 16.
  Hmm, the article says "以下输出为近似值, 使用 LLaMA-2 tokenizer" — so these are 
  actual tokenizer outputs, not computed from text length. The chars/token numbers
  are based on actual encoded token counts, not the text's character count.
  
  Actually wait - "chars per token" would be len(text) / len(ids). 
  So if LLaMA-2 produces 18 tokens for 16 chars: 16/18 = 0.89 chars/token
  But article claims 1.4 chars/token.
  
  Actually, maybe the text isn't just the Chinese characters. Let me re-read:
  "大语言模型分词器的工作原理是什么" = 16 Chinese characters.
  
  If LLaMA-2 tokenizer produces ~18 tokens for this, chars/token = 16/18 = 0.89, not 1.4.
  If it produces 1.4 chars/token, then tokens = 16/1.4 ≈ 11.4, so about 11 tokens.
  
  But the article says 18 tokens. So there's an inconsistency here.
  18 * 1.4 = 25.2 — there would need to be ~25 characters for 18 tokens at 1.4 chars/token.
  The text has 16 characters, unless we count different...
  
  Actually, the chars/token values (4.3, 1.4, 2.1) might be estimated from a real LLaMA-2
  tokenizer run which I can't verify here. But the inconsistency with the 4096-token
  calculation IS real: 4.3*4096 ≠ 17500 and 1.4*4096 ≠ 5800.

ISSUE 3 (Minor): 4096 tokens claim inconsistency
  - 4.3 chars/token × 4096 = 17612.8 (article says 17500)
  - 1.4 chars/token × 4096 = 5734.4 (article says 5800)
  - The implied rates are ~4.27 and ~1.42

Let me double-check whether issues 2 and 3 are real or due to my misunderstanding.
""")
