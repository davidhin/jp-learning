# %%
import os
import re
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


def insert_furigana(row, known_kanji):
    """Add furigana to unknown kanjis."""
    hiragana_full = r"[ぁ-ゟ]"
    replacements_raw = jph.get_furigana(row.ID, known_kanji)
    replacements = {}
    for rraw in replacements_raw.items():
        present_hira = re.findall(hiragana_full, rraw[0])
        temp_item_1 = rraw[0]
        temp_item_2 = rraw[1]
        for h in present_hira:
            # :TODO: This could have an edge case mistake in the future
            # e.g. if 手 is unknown in 手て, then it would remove both readings.
            temp_item_1 = re.sub(h, "", temp_item_1)
            temp_item_2 = re.sub(h, "", temp_item_2)
        replacements[temp_item_1] = temp_item_2
    ret = row.japanese
    for r in replacements.items():
        ret = ret.replace(r[0], r[1][:-1])
    return ret


df["japanese"] = df.apply(insert_furigana, known_kanji=known_kanji, axis=1)

df = df[["ID", "japanese", "english", "notes", "hira", "roman", "tags"]]
df.to_csv(jpl.outputs_dir() / "misa_anki.csv", index=0, header=None)
