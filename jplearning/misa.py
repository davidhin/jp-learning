# %%
import os
import re
from collections import defaultdict
from glob import glob

import pandas as pd

import jplearning as jpl
import jplearning.helpers as jph

# Get WK DF
wk_df = jph.get_vocab_df(os.getenv("WANIKANI"), type="kanji")
wk_df = wk_df[wk_df.srs_stage > 1]
known_kanji = set(wk_df.characters.tolist())


# %%
def sub_rks(text):
    """Generate hira/roman for jp sentence."""
    text = re.sub("\\<(.*?)\\>", "", text)
    return jph.read_kanji_sentence(text)


c = [
    (235, 172, 35),
    (184, 0, 88),
    (0, 140, 249),
    (0, 110, 0),
    (0, 187, 173),
    (209, 99, 230),
    (178, 69, 2),
    (255, 146, 135),
    (89, 84, 214),
    (0, 198, 248),
    (135, 133, 0),
    (0, 167, 108),
    (189, 189, 189),
]

all_pos = [
    "verb",
    "noun",
    "teform",
    "obj",
    "adv",
    "subject",
    "adbph",
    "conjunc",
    "det",
    "adj",
    "end",
]

df = pd.concat(
    [pd.read_csv(i) for i in sorted(glob(str(jpl.external_dir() / "misa/lesson*.csv")))]
)

# Sanity checks
for i in df.itertuples():
    count_la = i.japanese.count("<")
    count_ra = i.japanese.count(">")
    count_slash = i.japanese.count("/")
    markers = jph.get_all_in_tri_brack(i.japanese)
    for m in markers:
        assert m in all_pos + ["/"], "Unknown marker: {}".format(m)
    assert (i.japanese.count("<") + i.japanese.count(">")) % 2 == 0
    assert i.japanese.count("<") == i.japanese.count(">"), "Unmatched <> {}".format(
        i.japanese
    )
    assert i.japanese.count("<") / 2 == i.japanese.count(
        "/"
    ), "Unmatched </>: {}".format(i.japanese)

df.japanese = df.japanese.str.replace("</>", "</span>")
for it, pos in enumerate(all_pos):
    df.japanese = df.japanese.str.replace(pos, 'span style="color:rgb{}"'.format(c[it]))
df.notes = df.notes.str.replace("\\n", "<br />", regex=False)

hira_roman = df.japanese.apply(sub_rks)
df["hira"] = [i[0] for i in hira_roman]
df["roman"] = [i[1] for i in hira_roman]

# Filter known kanji
df["ID"] = df.japanese.apply(lambda x: re.sub("\\<(.*?)\\>", "", x))
df = df[["ID"] + list(df.columns[:-1])]

unknown_words = []


def insert_furigana(row, known_kanji):
    """Add furigana to unknown kanjis.

    GLOBAL: unknown words = list
    """
    replacements = jph.get_furigana(row.ID, known_kanji)
    ret = row.japanese
    for r in replacements.items():
        if r[0] not in ret:
            print(r)
        ret = ret.replace(r[0], r[1][:-1])
        unknown_words.append([r[1][:-1], row.english])
    kk_inc = jph.get_katakana(row.ID)
    if len(kk_inc) > 0:
        unknown_words.append(
            [
                "".join(kk_inc)
                + "[{}]".format(jph.read_kanji_sentence(kk_inc)[0][:-1]),
                row.english,
            ]
        )
    return ret


df["japanese"] = df.apply(insert_furigana, known_kanji=known_kanji, axis=1)

df = df[["ID", "japanese", "english", "notes", "hira", "roman", "tags"]]
df.to_csv(jpl.outputs_dir() / "misa_anki.csv", index=0, header=None)

# %% Generate kanji deck
unknown_kanji = []
for i in df.itertuples():
    kanji_list = jph.get_kanji(i.ID)
    for k in kanji_list:
        if k not in known_kanji:
            ex = i.ID + " -> " + i.english
            ex = ex.replace(
                k, '<span style="color:rgb(235, 172, 35)">{}</span>'.format(k)
            )
            unknown_kanji.append({"kanji": k, "example": ex})
ukdf = pd.DataFrame.from_records(unknown_kanji)
ukdf = ukdf.groupby("kanji").agg({"example": lambda x: "<br />".join(list(x))})
ukdf = ukdf.reset_index()
ukdf.to_csv(jpl.outputs_dir() / "misa_kanji_examples.csv", index=0, header=None)
wk_kanji = pd.read_parquet(jpl.external_dir() / "wanikani/kanji.parquet")
wk_kanji = wk_kanji[wk_kanji.characters.isin(ukdf.kanji)]
ukdf = ukdf.merge(wk_kanji, left_on="kanji", right_on="characters", how="outer").drop(
    columns=["characters", "pos", "srs_stage"]
)
ukdf = ukdf.fillna(0)
ukdf.subject_id = ukdf.subject_id.astype(int)

# Download WK data
for i in ukdf.subject_id:
    jph.download_subject(i)
    data = jph.load_subject(i)
    if "data" in data:
        for j in data["data"]["component_subject_ids"]:
            jph.download_subject(j)

# Join final kanji DF
cols = ["level", "meaning_mnemonic", "meaning_hint", "reading_mnemonic", "reading_hint"]
ukdf_wk = defaultdict(list)

level = []
for i in ukdf.itertuples():
    subject = jph.load_subject(i.subject_id)
    for c in cols:
        if i.subject_id == 0:
            ukdf_wk[c] += [None]
        else:
            ukdf_wk[c] += [subject["data"][c]]

for c in cols:
    ukdf[c] = ukdf_wk[c]

kanji_replacement_list = [
    "meaning_mnemonic",
    "meaning_hint",
    "reading_mnemonic",
    "reading_hint",
]
for kr in kanji_replacement_list:
    ukdf[kr] = ukdf[kr].str.replace("<kanji>", '<span style="color:#cf15b0">')
    ukdf[kr] = ukdf[kr].str.replace("</kanji>", "</span>")
    ukdf[kr] = ukdf[kr].str.replace("<radical>", '<span style="color:#1582cf">')
    ukdf[kr] = ukdf[kr].str.replace("</radical>", "</span>")


ukdf = ukdf.dropna()
kanji_level = ukdf[["kanji", "level"]].set_index("kanji").to_dict()["level"]

# %% Generate unknown word CSV
examples = (
    pd.DataFrame(unknown_words).drop_duplicates(subset=0).set_index(0).to_dict()[1]
)

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
word_df.to_csv(jpl.interim_dir() / "unknown_words.csv", index=0)
