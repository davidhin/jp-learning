# %%
import os

import pandas as pd
import requests

import jplearning as jpl
import jplearning.helpers as jph

# %% Get WK DF
# Only need to sync vocab once at the start, unless WaniKani updates its cards.
wk_df = jph.get_vocab_df(os.getenv("WANIKANI"), sync_vocab=False, type="kanji")
wk_df = wk_df[wk_df.srs_stage > 1]
known_kanji = set(wk_df.characters.tolist())

# %% Get Grammar Points
bpkey = os.getenv("BUNPRO")
bpresp = requests.get("https://bunpro.jp/api/user/{}/recent_items".format(bpkey))
bpresp = bpresp.json()
grammar_points = set([i["grammar_point"] for i in bpresp["requested_information"]])

# %% Get Bunpro Sentences
bpdf = pd.read_csv(jpl.external_dir() / "bunpro/bunpro.txt", sep="\t", header=None)
bpdf.columns = ["jp", "furi", "eng", "grammar", "tags"]
bpdf = bpdf.drop(columns=["tags", "furi"])
bpdf["used_kanji"] = bpdf.jp.apply(lambda x: set(jph.get_kanji(x)))
bpdf["should_learn"] = bpdf.apply(
    lambda x: x.used_kanji.issubset(known_kanji) and x.grammar in grammar_points, axis=1
)
bpdf = bpdf[bpdf.should_learn]
hira_roman = bpdf.jp.progress_apply(jph.read_kanji_sentence)
bpdf["hira"] = [i[0] for i in hira_roman]
bpdf["roman"] = [i[1] for i in hira_roman]

# %% Get Sentence DB (General)
sentence_db = jph.get_sentence_db().reset_index(drop=True)
sentence_db["used_kanji"] = sentence_db.jp.apply(lambda x: set(jph.get_kanji(x)))
sentence_db["should_learn"] = sentence_db.used_kanji.apply(
    lambda x: x.issubset(known_kanji)
)
sentence_db = sentence_db[sentence_db.should_learn]
sentence_db["jp_len"] = sentence_db.jp.apply(len)
sentence_db["kanji_len"] = sentence_db.used_kanji.apply(len)
sentence_db = sentence_db.groupby("jp").head(1)

# %% Sample sentences
sample = sentence_db.sort_values("jp_len").head(10)
hira_roman = sample.jp.progress_apply(jph.read_kanji_sentence)
sample["hira"] = [i[0] for i in hira_roman]
sample["roman"] = [i[1] for i in hira_roman]
sample["grammar"] = ""
sample["tags"] = sample["source"]

# %% Map grammar points
sgm = (
    pd.read_csv(jpl.external_dir() / "bunpro/sentence_grammar_mappings.csv")
    .set_index("jp")
    .join(sentence_db.set_index("jp"))
).reset_index()
hira_roman = sgm.jp.progress_apply(jph.read_kanji_sentence)
sgm["hira"] = [i[0] for i in hira_roman]
sgm["roman"] = [i[1] for i in hira_roman]
sgm["tags"] = sgm.source
sgm = sgm[["jp", "eng", "grammar", "hira", "roman", "tags"]]

# %% Save to anki
anki_bpdf = bpdf[["jp", "eng", "grammar", "hira", "roman"]].copy()
anki_bpdf["tags"] = "bunpro"
anki_sample = sample[["jp", "eng", "grammar", "hira", "roman", "tags"]].copy()
ankidf = pd.concat([anki_bpdf]).sort_values("grammar", ascending=0)
ankidf = ankidf.groupby("jp").head(1)
ankidf = ankidf.sort_values("jp")
ankidf.to_csv(jpl.outputs_dir() / "jp_anki.csv", index=0, header=None)

# %% Get Extra Grammar Point Examples
# grammar_point_list = sorted(list(grammar_points))
# sentence_db[
#     (sentence_db.jp.str.contains("の"))
#     # & (sentence_db.eng.str.contains("good"))
# ].sort_values("jp_len").head(100)
