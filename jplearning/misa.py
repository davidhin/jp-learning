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

all_pos = ["verb", "noun", "teform", "obj", "adv", "subject", "adbph", "det"]

df = pd.concat(
    [pd.read_csv(i) for i in glob(str(jpl.external_dir() / "misa/lesson*.csv"))]
)


df.japanese = df.japanese.str.replace("</>", "</span>")
for it, pos in enumerate(all_pos):
    df.japanese = df.japanese.str.replace(
        pos, 'span style=""color:rgb{}""'.format(c[it])
    )

hira_roman = df.japanese.apply(sub_rks)
df["hira"] = [i[0] for i in hira_roman]
df["roman"] = [i[1] for i in hira_roman]

# Filter known kanji
df["used_kanji"] = df.japanese.apply(lambda x: set(jph.get_kanji(x)))
df["should_learn"] = df.apply(lambda x: x.used_kanji.issubset(known_kanji), axis=1)
df = df[df.should_learn]
df = df[["japanese", "english", "notes", "hira", "roman", "tags"]]

df.to_csv(jpl.outputs_dir() / "misa_anki.csv", index=0, header=None)

# %%
