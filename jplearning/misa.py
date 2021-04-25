# %%
import os
import re
from glob import glob

import pandas as pd

import jplearning as jpl
import jplearning.helpers as jph
from jplearning.constants import ALL_POS, COLORS

# Get WK DF
wk_df = jph.get_vocab_df(os.getenv("WANIKANI"), type="kanji")
wk_df = wk_df[wk_df.srs_stage > 1]
known_kanji = set(wk_df.characters.tolist())
lesson_glob = sorted(glob(str(jpl.external_dir() / "misa/lesson*.csv")))
df = pd.concat([pd.read_csv(i) for i in lesson_glob])
jph.sanity_check_notes(df.japanese)

# Get extra information
df.japanese = df.japanese.str.replace("</>", "</span>")
for it, pos in enumerate(ALL_POS):
    df.japanese = df.japanese.str.replace(
        pos, 'span style="color:rgb{}"'.format(COLORS[it])
    )
df.notes = df.notes.str.replace("\\n", "<br />", regex=False)
hira_roman = df.japanese.apply(jph.rm_html_apply_rks)
df["hira"] = [i[0] for i in hira_roman]
df["roman"] = [i[1] for i in hira_roman]

# Create ID column with no HTML tags
df["ID"] = df.japanese.apply(lambda x: re.sub("\\<(.*?)\\>", "", x))
df = df[["ID"] + list(df.columns[:-1])]


def insert_furigana(row):
    """Add furigana to unknown kanjis."""
    ret = row.japanese
    for r in row.furigana.items():
        if r[0] not in ret:
            print(r)
        ret = ret.replace(r[0], r[1][:-1])
    return ret


# Insert furigana into japanese column
df["furigana"] = df.ID.apply(lambda x: jph.get_furigana(x, known_kanji))
df["japanese"] = df.apply(insert_furigana, axis=1)

# Get unknown words
unknown_words = []
for row in df.itertuples():
    for r in row.furigana.items():
        unknown_words.append([r[1][:-1], row.english])
    kk = jph.get_katakana_parts(row.ID)
    for k in kk:
        unknown_words.append([k + "[]", row.english])

# Save Misa Anki deck
df = df[["ID", "japanese", "english", "notes", "hira", "roman", "tags"]]
df["tags"] = df.apply(
    lambda x: x.tags + " unknown" if "[" in x.japanese else x.tags + " known", axis=1
)
df.to_csv(jpl.outputs_dir() / "misa_anki.csv", index=0, header=None)

# Get kanji levels
unknown_kanji = []
for i in df.itertuples():
    kanji_list = jph.get_kanji(i.ID)
    for k in kanji_list:
        if k not in known_kanji:
            unknown_kanji.append(k)
kanji_level = jph.assign_wklevel_to_kanji(unknown_kanji)

# Generate unknown word CSV
custom_kanji_cards = pd.read_csv(jpl.external_dir() / "custom_kanji_cards.csv")
custom_kanji_cards.to_csv(jpl.outputs_dir() / "misa_words.csv", index=0, header=0)
custom_mappings = (
    pd.read_csv(jpl.external_dir() / "custom_mappings.csv")
    .set_index("word")
    .to_dict()["root"]
)

# Interrim file to help add new unknown words
custom_exists = set(custom_kanji_cards.kanji)
uwdf = pd.DataFrame(unknown_words)
uwdf = uwdf.sort_values(1)
examples = uwdf.drop_duplicates(subset=0).set_index(0).to_dict()[1]
word_dict = []
for word in examples.keys():
    kanjiword = word.split("[")[0]
    romaji = word.split("[")[1][:-1]
    levels = {}
    for i in kanjiword:
        if i in kanji_level:
            levels[i] = kanji_level[i]
    word_dict.append(
        {
            "kanji": kanjiword,
            "romaji": romaji,
            "levels": levels,
            "example": examples[word],
        }
    )
word_df = pd.DataFrame.from_records(word_dict)
word_df["translation"] = ""
word_df["mnemonic"] = ""
word_df.kanji = word_df.kanji.apply(
    lambda x: custom_mappings[x] if x in custom_mappings else x
)
word_df = word_df[~word_df.kanji.isin(custom_exists)]
word_df.to_csv(jpl.interim_dir() / "no_custom_word_meaning.csv", index=0)

# Custom Mappings
dictforms = {}
for row in df.itertuples():
    sentence = row.ID
    dictform_tups = jph.get_unknown_dictform_words(sentence, known_kanji)
    dictforms = {**dictforms, **dictform_tups}
dictforms = pd.DataFrame.from_dict(dictforms.items()).drop_duplicates()
dictforms.columns = ["word", "root"]
dictforms = dictforms[dictforms.word != dictforms.root]
dictforms.to_csv(jpl.interim_dir() / "auto_mappings.csv", index=0)

# %%
# ukwdf = []
# for row in df.itertuples():
#     sentence = row.ID
#     dictform_tups = list(jph.get_unknown_dictform_words(sentence, known_kanji).items())
#     ukwdf += [[i[1], row.ID, row.english] for i in dictform_tups]
#     kk = jph.get_katakana_parts(row.ID)
#     ukwdf += [[k, row.ID, row.english] for k in kk]
# ukwdf = pd.DataFrame(ukwdf, columns=["word", "japanese", "example"])
# ukwdf = ukwdf.drop_duplicates(subset="word")
# ukwdf[["hira"]] = [i[0] for i in ukwdf.word.apply(jph.read_kanji_sentence)]
# ukwdf["levels"] = ukwdf.word.apply(
#     lambda x: dict([[i, kanji_level[i]] for i in x if i in kanji_level])
# )
# ukwdf["translation"] = ""
# ukwdf["mnemonic"] = ""
# ukwdf[
#     ["word", "hira", "levels", "example", "japanese", "translation", "mnemonic"]
# ].to_csv(jpl.interim_dir() / "no_custom_word_meaning.csv", index=0)

# # %%
# custom_mnemonics = pd.read_csv(jpl.external_dir() / "custom_kanji_cards.csv")
# custom_mnemonics
# ukwdf[~ukwdf.word.isin(custom_exists)]
# custom_kanji_cards[~custom_kanji_cards.kanji.isin(set(ukwdf.word))]
