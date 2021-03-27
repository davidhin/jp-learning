# %%
import os

import pandas as pd

import jplearning as jpl
import jplearning.helpers as jph

# %% Get WK DF
# Only need to sync vocab once at the start, unless WaniKani updates its cards.
wk_df = jph.get_vocab_df(os.getenv("WANIKANI"), sync_vocab=False, type="kanji")
wk_df = wk_df[wk_df.srs_stage > 1]
known_kanji = set(wk_df.characters.tolist())

# %% Get Bunpro Sentences
bpdf = pd.read_csv(jpl.external_dir() / "bunpro/bunpro.txt", sep="\t", header=None)
bpdf.columns = ["jp", "furi", "eng", "grammar", "tags"]
bpdf = bpdf.drop(columns=["tags", "furi"])
bpdf["used_kanji"] = bpdf.jp.apply(lambda x: set(jph.get_kanji(x)))
bpdf["should_learn"] = bpdf.apply(lambda x: x.used_kanji.issubset(known_kanji), axis=1)
bpdf = bpdf[bpdf.should_learn]
hira_roman = bpdf.jp.progress_apply(jph.read_kanji_sentence)
bpdf["hira"] = [i[0] for i in hira_roman]
bpdf["roman"] = [i[1] for i in hira_roman]

# %%
bpdf[bpdf.grammar.str.contains("Verbs")]

# %%
