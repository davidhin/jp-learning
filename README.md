# Japanese Learning

Used to help David create Anki decks for studying Japanese
Setup:

```
pip install -e .
```

Move the following files into `storage/external` (https://tatoeba.org/eng/downloads):

- tatoeba_eng.tsv
- tatoeba_jp.tsv
- tatoeba_links.csv

## Process of adding new content

1. Watch Japanese Ammo with Misa lesson and create lesson_X.csv in `storage/external/misa`
2. Run `python jplearning/misa.py`
3. Using generated interim files for assistance, update custom_mnemonics.csv and custom_mappings.csv
4. Run `python jplearning/misa.py`
5. Import `storage/outputs` into anki accordingly
