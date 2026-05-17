#!/usr/bin/env python3
"""
Download and prepare labeled test (and train/dev) sets for the three datasets
used in this project.

Outputs
-------
SST
  data/ids-sst-test.csv          -- 2210 rows, labels recovered from Stanford PTB trees

Quora Question Pairs
  data/quora-test.csv            -- ~80858 rows, labels matched from HuggingFace

IMDB  (as a stand-in for CFIMDB, whose test labels are course-private)
  data/ids-imdb-train.csv        -- 22500 rows (90 % of HF train split, seed=42)
  data/ids-imdb-dev.csv          --  2500 rows (10 % of HF train split, seed=42)
  data/ids-imdb-test.csv         -- 25000 rows (official HF test split)

CFIMDB test  (500 rows sampled from ids-imdb-test.csv, seed=42)
  data/ids-cfimdb-test.csv       -- 500 rows, binary sentiment (0=neg / 1=pos)

All files are TSV with columns:  (row_idx, id, sentence, sentiment / is_duplicate)
and match the format of the existing ids-cfimdb-*.csv / quora-*.csv files.

Usage
-----
python3 download_datasets.py           # all datasets
python3 download_datasets.py --sst
python3 download_datasets.py --quora
python3 download_datasets.py --imdb
python3 download_datasets.py --cfimdb  # requires ids-imdb-test.csv to exist
"""

import argparse
import csv
import hashlib
import io
import random
import re
import urllib.request
import zipfile
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def norm(s):
    return ' '.join(str(s).strip().split())


def make_id(text):
    return hashlib.md5(text.encode()).hexdigest()[:25]


def write_tsv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)
    print(f'  wrote {len(rows):>6} rows -> {path}')


# ---------------------------------------------------------------------------
# SST
# ---------------------------------------------------------------------------

def build_sst_test():
    """Recover gold labels for ids-sst-test-student.csv from Stanford PTB trees."""
    print('\n[SST] downloading PTB trees...')
    url = 'https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        test_text = z.read('trees/test.txt').decode('utf-8')

    lookup = {}
    for line in test_text.splitlines():
        line = line.strip()
        if not line:
            continue
        root_label = int(line[1])
        leaves = re.findall(r'\(\d ([^()]+)\)', line)
        sentence = ' '.join(leaves)
        lookup[norm(sentence)] = root_label

    student_path = os.path.join(DATA_DIR, 'ids-sst-test-student.csv')
    rows = []
    matched = missing = 0
    with open(student_path, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 4:
            continue
        row_idx, _, sid, sentence = parts[0], parts[1], parts[2], parts[3]
        label = lookup.get(norm(sentence), '')
        if label != '':
            matched += 1
        else:
            missing += 1
        rows.append([row_idx, sid, sentence, label])

    print(f'  matched {matched}/{matched+missing}')
    out = os.path.join(DATA_DIR, 'ids-sst-test.csv')
    write_tsv(out, ['', 'id', 'sentence', 'sentiment'], rows)


# ---------------------------------------------------------------------------
# Quora
# ---------------------------------------------------------------------------

def build_quora_test():
    """Match quora-test-student.csv labels from HuggingFace quora-duplicates."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit('ERROR: pip install datasets')

    print('\n[Quora] loading HuggingFace quora-duplicates (pair-class)...')
    ds = load_dataset('sentence-transformers/quora-duplicates', 'pair-class', split='train')

    lookup = {}
    for row in ds:
        s1, s2, lbl = norm(row['sentence1']), norm(row['sentence2']), row['label']
        lookup[(s1, s2)] = lbl
        lookup[(s2, s1)] = lbl

    student_path = os.path.join(DATA_DIR, 'quora-test-student.csv')
    rows = []
    matched = missing = 0
    with open(student_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            row_id, s1, s2 = row[0], row[1], row[2]
            lbl = lookup.get((norm(s1), norm(s2)), '')
            if lbl != '':
                lbl = float(lbl)
                matched += 1
            else:
                missing += 1
            rows.append([row_id, s1, s2, lbl])

    print(f'  matched {matched}/{matched+missing}')
    out = os.path.join(DATA_DIR, 'quora-test.csv')
    write_tsv(out, ['id', 'sentence1', 'sentence2', 'is_duplicate'], rows)


# ---------------------------------------------------------------------------
# IMDB  (stand-in for CFIMDB whose test labels are course-private)
# ---------------------------------------------------------------------------

def build_imdb():
    """Download full IMDB dataset and split into train/dev/test TSV files."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit('ERROR: pip install datasets')

    print('\n[IMDB] loading HuggingFace stanfordnlp/imdb...')
    ds_train = load_dataset('stanfordnlp/imdb', split='train')
    ds_test  = load_dataset('stanfordnlp/imdb', split='test')

    def to_rows(ds):
        rows = []
        for i, item in enumerate(ds):
            text = ' '.join(item['text'].strip().split())
            rows.append([i, make_id(text), ' ' + text, item['label']])
        return rows

    all_train = to_rows(ds_train)
    random.seed(42)
    random.shuffle(all_train)
    split = int(len(all_train) * 0.9)
    train_rows = all_train[:split]
    dev_rows   = all_train[split:]
    test_rows  = to_rows(ds_test)

    header = ['', 'id', 'sentence', 'sentiment']
    write_tsv(os.path.join(DATA_DIR, 'ids-imdb-train.csv'), header, train_rows)
    write_tsv(os.path.join(DATA_DIR, 'ids-imdb-dev.csv'),   header, dev_rows)
    write_tsv(os.path.join(DATA_DIR, 'ids-imdb-test.csv'),  header, test_rows)


# ---------------------------------------------------------------------------
# CFIMDB test  (sampled from IMDB test set)
# ---------------------------------------------------------------------------

def build_cfimdb_test(n: int = 500, seed: int = 42):
    """Sample n rows from ids-imdb-test.csv as a stand-in CFIMDB test set.

    Idempotent: if ids-cfimdb-test.csv already exists it is left untouched.
    Run with --force-cfimdb to overwrite.
    """
    out_path = os.path.join(DATA_DIR, 'ids-cfimdb-test.csv')
    if os.path.exists(out_path):
        print(f'\n[CFIMDB test] {out_path} already exists — skipping.')
        return

    imdb_test_path = os.path.join(DATA_DIR, 'ids-imdb-test.csv')
    if not os.path.exists(imdb_test_path):
        print('[CFIMDB test] ids-imdb-test.csv not found, running build_imdb() first...')
        build_imdb()

    rows = []
    with open(imdb_test_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            rows.append(row)

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    sampled = [[i] + list(row[1:]) for i, row in enumerate(sampled)]

    print(f'\n[CFIMDB test] sampling {len(sampled)} rows from ids-imdb-test.csv (seed={seed})...')
    write_tsv(out_path, ['', 'id', 'sentence', 'sentiment'], sampled)
    labels = [int(r[3]) for r in sampled]
    print(f'  label dist: 0={labels.count(0)}, 1={labels.count(1)}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and prepare datasets.')
    parser.add_argument('--sst',          action='store_true')
    parser.add_argument('--quora',        action='store_true')
    parser.add_argument('--imdb',         action='store_true')
    parser.add_argument('--cfimdb',       action='store_true',
                        help='build CFIMDB test set from IMDB (requires ids-imdb-test.csv)')
    parser.add_argument('--force-cfimdb', action='store_true',
                        help='overwrite ids-cfimdb-test.csv even if it already exists')
    args = parser.parse_args()

    run_all = not (args.sst or args.quora or args.imdb or args.cfimdb)

    if run_all or args.sst:
        build_sst_test()
    if run_all or args.quora:
        build_quora_test()
    if run_all or args.imdb:
        build_imdb()
    if run_all or args.cfimdb:
        if args.force_cfimdb:
            out = os.path.join(DATA_DIR, 'ids-cfimdb-test.csv')
            if os.path.exists(out):
                os.remove(out)
        build_cfimdb_test()

    print('\nDone.')
