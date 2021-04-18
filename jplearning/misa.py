# %%
import json
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


def insert_furigana(row, known_kanji):
    """Add furigana to unknown kanjis."""
    replacements = jph.get_furigana(row.ID, known_kanji)
    ret = row.japanese
    for r in replacements.items():
        if r[0] not in ret:
            print(r)
        ret = ret.replace(r[0], r[1][:-1])
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
ukdf.to_csv(jpl.outputs_dir() / "misa_kanji.csv", index=0, header=0)

# # %% Get radicals
# radicals = []
# wkuser = jph.get_wk_user()
# wklevel = wkuser["data"]["level"]
# for p in glob(str(jpl.external_dir() / "wanikani/*.json")):
#     with open(p) as f:
#         data = json.load(f)
#     if "0.json" in p:
#         continue
#     if data["object"] != "radical":
#         continue
#     if data["data"]["level"] <= wklevel:
#         continue
#     rel_data = {}
#     rel_data["id"] = data["id"]
#     rel_data["level"] = data["data"]["level"]
#     rel_data["slug"] = data["data"]["slug"]
#     rel_data["character"] = data["data"]["characters"]
#     if len(data["data"]["character_images"]) > 3:
#         rel_data["image"] = data["data"]["character_images"][4]["url"]
#     else:
#         rel_data["image"] = None
#     rel_data["meanings"] = ", ".join(
#         [mean["meaning"] for mean in data["data"]["meanings"]]
#     )
#     rel_data["meaning_mnemonic"] = data["data"]["meaning_mnemonic"]
#     radicals.append(rel_data)

# # %% Get radical DF
# radical_df = pd.DataFrame.from_records(radicals).sort_values("level")
