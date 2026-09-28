# Trap annotation guide (two annotators and the lead)

This guide is for the two annotators and the lead who decide which trap candidates become the trap set (design doc
`Laya Record-Normalisation PoC.md` §5.3 "Trap set construction", §7.4.5). Each annotator marks 700 rows (about a
day of work at half a minute per row), then the lead decides the rows where they disagree. The data engineer (DE)
makes the copies, runs the check and runs the merge.

## 1. What a trap is, and why it matters

A trap is a real place whose **name misleads** about its category, while its FSQ category is right: a canteen called
"... Hospital ..." that FSQ files under *Dining and Drinking*, a bakery on Bank Street, a shop called "Hotel ...". It
can also be a place with several level-1 categories (a multi-category brand). Trap accuracy (§5.11) shows whether a
model reads the whole record or just reacts to a word in the name. Decision criterion 4 (§5.12) needs it: Laya's trap
accuracy must be at least that of the best baseline.

The frozen build set aside 700 candidates, 100 per pattern, and removed them (and every row sharing their dedup key)
from every split, so no model trained on them. Every Phase 3 and 4 run has already scored all 700. The annotation
only decides **which candidates count**; nothing is retrained. The target is **150-300 kept items**.

**The candidates are FSQ rows.** Keep them on the team's machines: do not email them, upload them to online services
(online spreadsheets included) or commit them (`data/` is gitignored). The FSQ terms and `data/NOTICE_FSQ.txt` apply.

## 2. The file

`data/trap_candidates.csv`, one row per candidate, in the layout `python -m laya_poc.traps` defines:

| Column | Meaning | Who edits it |
|---|---|---|
| `pattern` | Why the row is a candidate (table below) | nobody |
| `fsq_place_id` | The place's id: every later step joins on it | nobody, ever |
| `key` | The dedup key (normalised name, locality, country) | nobody |
| `name`, `address`, `locality`, `region`, `postcode`, `admin_region`, `post_town`, `po_box`, `country`, `tel`, `website`, `email`, `facebook_id`, `instagram`, `twitter` | The record's fields: what the models saw | nobody |
| `label` | The FSQ level-1 category: the gold answer. For a multi-category place, the level-1 of its **first listed** category | nobody |
| `label_key` | The same category as the models' option key (`dining`, `arts`, `health`, ...) | nobody |
| `multi` | 1 when the place has several level-1 categories (reported separately), else 0 | nobody |
| `n_l1`, `l1s` | How many level-1 categories the place has, and which | nobody |
| `keep_a1` | Annotator 1's mark: 1 keep, 0 drop | annotator 1 |
| `keep_a2` | Annotator 2's mark: 1 keep, 0 drop | annotator 2 |
| `keep_lead` | The lead's decision, on the rows where the annotators disagree | the lead |
| `note` | Optional, short: why (each person may add to it) | everyone |

| `pattern` | The name contains | The label is not |
|---|---|---|
| `health_word_not_health` | hospital, clinic, pharmacy, dental, surgery | Health and Medicine |
| `bank_word` | bank | Business and Professional Services |
| `church_school_word` | church, chapel, school, college, abbey | Community and Government |
| `museum_theatre_word` | museum, theatre, theater, gallery, cinema | Arts and Entertainment |
| `park_garden_word` | park, garden, beach, lake | Landmarks and Outdoors |
| `station_hotel_word` | station, hotel, airport | Travel and Transportation |
| `multi_category` | (any name) the place has more than one level-1 category (`multi` = 1) | - |

## 3. The keep rule

Mark a row **1 (keep) only if both hold**; otherwise **0 (drop)**:

1. **The FSQ label is plausibly correct.** Taken together, the row's fields (name, address, website, email, social
   handles) are consistent with `label`: a reasonable person could file this place there. Judge from the row and
   general knowledge; do not look the place up. If the fields contradict the label (an actual hospital filed under
   *Dining and Drinking*), drop it: that row would measure FSQ's label noise, not the model.
2. **The name misleads.** Read on its own, the name points to a different category than `label`, usually the one of
   the pattern's word. If the rest of the name settles the category anyway ("Lakeside Grill" for a restaurant), the
   name does not mislead: drop it.

Examples (invented, not from the data):

| `pattern` | Name | `label` | Mark | Why |
|---|---|---|---|---|
| `health_word_not_health` | Riverside Hospital Canteen | Dining and Drinking | 1 | A canteen in a hospital: the label fits, the name says hospital |
| `health_word_not_health` | St Anne's Hospital, website `stanneshospital.example` | Dining and Drinking | 0 | Nothing supports Dining: most likely a label error |
| `bank_word` | Bank Street Bakery | Retail | 1 | A bakery; "Bank" is the street |
| `church_school_word` | The Old School House (a pub) | Dining and Drinking | 1 | A pub in a former school |
| `park_garden_word` | Lakeside Grill | Dining and Drinking | 0 | "Grill" already says Dining: the name does not mislead |
| `station_hotel_word` | Hotel Chocolat | Retail | 1 | A chocolate shop chain |
| `multi_category` | Northgate Brewery Tap, `l1s` Dining and Drinking; Retail | Retail | 1 | Its first category is the brewery shop; "Tap" points to Dining, one of its other categories |
| `multi_category` | Sunny Café, `l1s` Dining and Drinking; Retail | Dining and Drinking | 0 | The name plainly says the gold class |

**Multi-category items** (`multi` = 1; §5.3 step 4). The gold is fixed: the level-1 of the place's first listed
category, in `label` (`l1s` lists them all, alphabetically). Apply the same rule: keep the item when that gold is a
plausible category for the place and the name points elsewhere, to one of its other categories (`l1s`) or to none of
them. They are reported separately (trap accuracy "multi"), so judge them as strictly as the rest.

When unsure, mark 0 and write a short note. If you both mark 0, the row is dropped without review; the lead only sees
the rows you disagree on.

## 4. Step by step

1. **The DE makes one copy per annotator.** Nobody opens `data/trap_candidates.csv` itself in a spreadsheet: the
   metrics map every scored candidate to its place by that file's row order and checksum, so it must stay exactly as
   the build wrote it.

   ```bash
   cp data/trap_candidates.csv data/trap_candidates_a1.csv   # annotator 1 (and the lead's keep_lead)
   cp data/trap_candidates.csv data/trap_candidates_a2.csv   # annotator 2
   ```

2. **Each annotator marks their own copy, independently.** Annotator 1 fills `keep_a1` (and `note`) in
   `data/trap_candidates_a1.csv`; annotator 2 fills `keep_a2` (and `note`) in `data/trap_candidates_a2.csv`. Every
   row gets a mark. Do not look at the other file or discuss rows until both are done.
3. **The DE checks the pair.** The check reads both files exactly as the merge does and prints how many rows each
   annotator marked, the agreement and Cohen's kappa on the rows both marked, the kept count so far, and every row
   the lead must decide, by its spreadsheet row number in annotator 1's file:

   ```bash
   .venv/Scripts/python tools/p5_trap_check.py data/trap_candidates_a1.csv data/trap_candidates_a2.csv
   ```

4. **The lead decides the disagreements.** For each listed row, the lead writes 1 or 0 in `keep_lead` of
   `data/trap_candidates_a1.csv` (only there: the merge refuses two different lead decisions) and may add to `note`.
   Re-run the check until it prints `ready for traps merge`. Keep its statistics (agreement, kappa, kept count) for
   the Phase 7 memo.
5. **The DE merges.** Kept = `keep_lead` 1, or, where the lead left it empty, both annotators 1:

   ```bash
   .venv/Scripts/python -m laya_poc.traps merge --candidates data/trap_candidates_a1.csv data/trap_candidates_a2.csv --out data/trap.jsonl --data-dir data
   ```

   The merge refuses a pair with unmarked rows or undecided disagreements, re-checks that no kept place is in any
   split, takes every field value from `data/pool.parquet` (so a spreadsheet cannot change what is scored), drops
   kept places outside the evaluation countries (and counts them), and prints the kept count per pattern and `multi`.
   It warns when the count is outside 150-300 (config `data.trap_target`).
6. **Then Phase 5 recomputes trap accuracy** from the saved predictions (`docs/phase5_runbook.md` Step 4). Keep both
   annotated copies unchanged after the merge: they are the record of the annotation (and of its statistics).

**About the target.** 150-300 is a target, not a quota: do not keep weak items to reach 150 or drop good ones to stay
under 300. If the kept set ends outside the range, the lead decides with the team and the memo says so (fewer items
mean a wider uncertainty on trap accuracy).

## 5. Values

- `keep_a1`, `keep_a2`, `keep_lead`: **1** (keep) or **0** (drop). Nothing else: "yes", "x", "TRUE" or text are
  refused (a spreadsheet's `1.0` / `0.0` is accepted).
- Every row needs both annotators' marks, or a lead decision.
- `keep_lead` is for disagreements. If set on any row, it overrides both annotators on that row.
- `note`: free text; commas are fine (the cell is quoted), avoid line breaks.

## 6. Spreadsheet pitfalls: encoding, sorting, ids

The file is UTF-8 with a BOM (byte-order mark), so accents and non-Latin scripts (Thai, Japanese) display correctly.
Spreadsheets also convert values when they open a CSV: postcodes lose their leading zeros, phone numbers and
`facebook_id` turn into numbers in scientific notation, and an `fsq_place_id` made of digits with one `e` becomes a
number. The merge reads field values from `pool.parquet`, so mangled fields do no harm, but a changed `fsq_place_id`
breaks the join (`... not in the pool`). So:

- **Open the file with every column as text.** Excel: *Data → From Text/CSV*, *File origin: 65001: Unicode (UTF-8)*,
  *Transform Data*, select all columns, *Data Type: Text*, *Close & Load*. LibreOffice Calc: *Open*, in the Text
  Import dialog *Character set: Unicode (UTF-8)*, *Separated by: Comma*, select all columns, *Column type: Text*.
- **Save it as CSV, same name.** Excel: *CSV UTF-8 (Comma delimited) (\*.csv)*. LibreOffice: *Keep current format*,
  *Character set: Unicode (UTF-8)*, field delimiter `,`, string delimiter `"`. Never save as `.xlsx` only, and keep
  the header row.
- **Do not sort or reorder rows**, and do not delete, insert or move rows or columns. To focus on one pattern, use a
  filter to view rows only, and clear it before saving. (The merge matches the two files by `pattern` and
  `fsq_place_id`, but the check's row numbers, the lead's work and the order of `data/trap.jsonl` all assume the
  original order; the check warns when the files' orders differ.)
- **Edit only your own columns**: `keep_a1` or `keep_a2` and `note`; the lead `keep_lead` and `note`.
- If a copy is damaged (rows lost or changed), make a fresh copy from `data/trap_candidates.csv` and move the marks
  over by `fsq_place_id`; the check says when the two files no longer hold the same candidates.
