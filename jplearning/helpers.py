"""Get wanikani vocab and build sample sentences."""
import json
import os
import re
from typing import List

import MeCab
import pandas as pd
import pykakasi
import requests
from tqdm import tqdm
from wanikani_api.client import Client

import jplearning as jpl
from jplearning.constants import ALL_POS

tqdm.pandas()


def get_all_in_tri_brack(s: str) -> list:
    """Extract all words inside triangular brackets in a string."""
    return re.findall("\\<(.*?)\\>", s)


def get_furigana(text, known_kanji={}):
    """Return furigana replacements."""
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    replacements = {}
    suffix_removal = ["て", "で", "く", "か"]
    for item in result:
        if not set(get_kanji(item["orig"])).issubset(known_kanji):
            if len(item["orig"]) > 1:
                for sr in suffix_removal:
                    if item["orig"][-1] == sr and item["hira"][-1] == sr:
                        print("Suffix {}: {} {}".format(sr, item["orig"], item["hira"]))
                        item["orig"] = item["orig"][:-1]
                        item["hira"] = item["hira"][:-1]
            furi = "{}[{}] ".format(item["orig"], item["hira"])
            replacements[item["orig"]] = furi
    return replacements


def extract_unicode_block(unicode_block, string):
    """Extract and returns all texts from a unicode block from string argument."""
    return re.findall(unicode_block, string)


def remove_unicode_block(unicode_block, string):
    """Remove all chaacters from a unicode block and returns all remaining texts."""
    return re.sub(unicode_block, "", string)


def get_kanji(text):
    """Get Kanji from text."""
    return extract_unicode_block(r"[㐀-䶵一-鿋豈-頻]", text)


def get_katakana(text):
    """Get Katakana from text."""
    return extract_unicode_block(r"[゠-ヿ]", text)


def read_kanji_sentence(text):
    """Get hiragana/romaji from kanji sentence."""
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    hira = ""
    roman = ""
    for item in result:
        hira += item["hira"] + " "
        roman += item["hepburn"] + " "
    return [hira, roman]


def rm_html_apply_rks(text):
    """Remove html tags before applying read_kanji_sentence()."""
    text = re.sub("\\<(.*?)\\>", "", text)
    return read_kanji_sentence(text)


def get_vocab_df(api_key, sync_vocab=False, type="kanji"):
    """Get vocab dataframe.

    Example:
        subject_id        meanings characters  readings        pos  srs_stage
    0        2467           [One]          一      [いち]  [numeral]          8
    1        2468     [One Thing]         一つ     [ひとつ]  [numeral]          8
    2        2469         [Seven]          七  [なな, しち]  [numeral]          8
    3        2470  [Seven Things]         七つ     [ななつ]  [numeral]          8
    4        2471          [Nine]          九  [きゅう, く]  [numeral]          8

    Args:
        api_key (str): wanikani api key
        sync_vocab (bool, optional): Fetch and cache vocab. Defaults to False.

    Returns:
        [pandas df]: See example.
    """
    # Get Vocab
    wkpath = jpl.get_dir(jpl.external_dir() / "wanikani")
    client = Client(api_key)
    if sync_vocab:
        all_vocabulary = client.subjects(types=type, fetch_all=True)
        vocab_db = []
        for vocab in all_vocabulary:
            row = []
            row.append(vocab.id)
            row.append([i.meaning for i in vocab.meanings])
            row.append(vocab.characters)
            row.append([i.reading for i in vocab.readings])
            if type == "vocabulary":
                row.append(vocab.parts_of_speech)
            else:
                row.append("")
            vocab_db.append(row)
        vocab_csv = pd.DataFrame(
            vocab_db,
            columns=["subject_id", "meanings", "characters", "readings", "pos"],
        )
        vocab_csv["srs_stage"] = 0
        vocab_csv.to_parquet(
            wkpath / "{}.parquet".format(type), index=0, compression="gzip",
        )
    else:
        vocab_csv = pd.read_parquet(wkpath / "{}.parquet".format(type))

    # Get SRS Scores
    assignments = client.assignments(subject_types=type, fetch_all=True)
    for assignment in assignments:
        vocab_csv.loc[
            vocab_csv.subject_id == assignment.subject_id, "srs_stage"
        ] = assignment.srs_stage

    return vocab_csv


def filter_pos(df, str_filter):
    """Filter dataframe by part-of-speech.

    Args:
        df (pandas df): DF returned from get_vocab_df()
        str_filter (str): Filter pos column

    Returns:
        [pandas df]: Filtered DF
    """
    res = []
    for i in df.itertuples():
        if pd.Series([str_filter in j for j in i.pos]).any():
            res.append(i)
    return pd.DataFrame(res)


def exact_pos(df, str_filter):
    """Filter dataframe by part-of-speech exactly.

    Args:
        df (pandas df): DF returned from get_vocab_df()
        str_filter (str): Filter pos column

    Returns:
        [pandas df]: Filtered DF
    """
    res = []
    for i in df.itertuples():
        if str_filter == i.pos[0]:
            res.append(i)
    return pd.DataFrame(res)


def filter_meaning(df, str_filter):
    """Filter dataframe by meaning.

    Args:
        df (pandas df): DF returned from get_vocab_df()
        str_filter (str): Filter pos column

    Returns:
        [pandas df]: Filtered DF
    """
    res = []
    for i in df.itertuples():
        if pd.Series([str_filter in j for j in i.meanings]).any():
            res.append(i)
    return pd.DataFrame(res)


def extract_vars(str):
    """Extract all strings between square brackets."""
    return re.findall(r"\[([A-Za-z0-9_]+)\]", str)


def get_replacement_dict(template_str, pos_dict):
    """Generate Replacement Dict given template str."""
    particles = extract_vars(template_str)
    replacement_dict = {}
    for p in particles:
        pos, _ = p.split("_")
        try:
            row_sample = pos_dict[pos].sample(1)
            replacement_dict[p] = [
                row_sample.meanings.item()[0],
                row_sample.characters.item(),
            ]
        except Exception as e:
            print(e)
    return replacement_dict


def get_sentence_db():
    """Get sentences from ./data folder."""
    # Read example sentences - Core6k
    core6k = pd.read_csv(
        jpl.external_dir() / "bunpro/core6k.txt", sep="\t", header=None
    )
    core6k[5] = core6k[5].str.replace("<b>", "")
    core6k[5] = core6k[5].str.replace("</b>", "")
    core6k[5] = core6k[5].apply(lambda x: re.sub("[\\(\\[].*?[\\)\\]]", "", x))
    core6k = core6k[[5, 6]]
    core6k.columns = ["jp", "eng"]
    core6k["source"] = "core6k"

    # Read example sentences - Tatoeba
    def get_tatoeba():
        tatoeba_jp = pd.read_csv(
            jpl.external_dir() / "bunpro/tatoeba_jp.tsv", sep="\t", header=None
        )
        tatoeba_eng = pd.read_csv(
            jpl.external_dir() / "bunpro/tatoeba_eng.tsv", sep="\t", header=None
        )
        tatoeba_links = pd.read_csv(
            jpl.external_dir() / "bunpro/tatoeba_links.csv", sep="\t", header=None
        )

        # Filter eng translations
        tatoeba_links = tatoeba_links[tatoeba_links[1].isin(set(tatoeba_eng[0]))]

        # Filter jp translations
        tatoeba_links = tatoeba_links[tatoeba_links[0].isin(set(tatoeba_jp[0]))]

        # Filter Eng links
        tatoeba_eng = tatoeba_eng[tatoeba_eng[0].isin(set(tatoeba_links[1]))]
        tatoeba_eng_dict = tatoeba_eng.set_index(0).to_dict()[2]

        # Get links
        tatoeba_links[1] = tatoeba_links[1].progress_apply(
            lambda x: tatoeba_eng_dict[x]
        )
        tatoeba_links_dict = tatoeba_links.groupby(0)[1].apply(list).to_dict()

        def jpid_to_eng(jpid):
            if jpid in tatoeba_links_dict:
                return tatoeba_links_dict[jpid][0]
            return ""

        tatoeba_jp[3] = tatoeba_jp[0].progress_apply(jpid_to_eng)
        tatoeba_jp.columns = [0, 1, "jp", "eng"]
        tatoeba_jp = tatoeba_jp[["jp", "eng"]]
        tatoeba_jp["source"] = "tatoeba"
        return tatoeba_jp

    totoeba = get_tatoeba()

    # Read example sentences - jomako
    # https://ankiweb.net/shared/info/1498427305
    # https://docs.google.com/spreadsheets/d/1ukDIWSkh_xvpppPbgs1nUR2kaEwFaWlsJgZUlb9LuTs/edit#gid=74793468
    jomako = pd.read_csv(jpl.external_dir() / "bunpro/jomako.csv")[["jp", "eng"]]
    jomako["source"] = "jomako"

    return pd.concat([totoeba, core6k, jomako])


def download_subject(id: int, verbose: int = 0):
    """Download subject info from wanikani."""
    jpl.get_dir(jpl.external_dir() / "wanikani")
    if os.path.exists(jpl.external_dir() / "wanikani/{}.json".format(id)):
        if verbose > 0:
            print("Already downloaded {}".format(id))
        return
    data = requests.get(
        "https://api.wanikani.com/v2/subjects/{}".format(id),
        headers={"Authorization": "Bearer {}".format(os.getenv("WANIKANI"))},
    )
    with open(jpl.external_dir() / "wanikani/{}.json".format(id), "w") as outfile:
        json.dump(data.json(), outfile, ensure_ascii=False)


def load_subject(id: int):
    """Load json."""
    with open(jpl.external_dir() / "wanikani/{}.json".format(id)) as f:
        data = json.load(f)
    return data


def get_wk_user():
    """Get WK User info."""
    data = requests.get(
        "https://api.wanikani.com/v2/user",
        headers={"Authorization": "Bearer {}".format(os.getenv("WANIKANI"))},
    )
    return data.json()


def assign_wklevel_to_kanji(kanjis: list):
    """Return a dict of wklevel assigned to a list of given kanji.

    Example:
    assign_wklevel_to_kanji(['板', '寝', '寝'])
    >>> {'板': 29, '寝': 22}

    Args:
        kanjis (list): A list of kanji.
    """
    ukdf = pd.DataFrame(kanjis, columns=["kanji"])
    wk_kanji = pd.read_parquet(jpl.external_dir() / "wanikani/kanji.parquet")
    wk_kanji = wk_kanji[wk_kanji.characters.isin(ukdf.kanji)]
    ukdf = ukdf.merge(
        wk_kanji, left_on="kanji", right_on="characters", how="outer"
    ).drop(columns=["characters", "pos", "srs_stage", "meanings", "readings"])
    ukdf = ukdf.fillna(0)
    ukdf.subject_id = ukdf.subject_id.astype(int)

    # Download WK data
    for i in ukdf.subject_id:
        download_subject(i)
        data = load_subject(i)
        if "data" in data:
            for j in data["data"]["component_subject_ids"]:
                download_subject(j)

    # Join final kanji DF
    ukdf["level"] = ukdf.apply(
        lambda x: load_subject(x.subject_id)["data"]["level"]
        if x.subject_id != 0
        else -1,
        axis=1,
    )

    ukdf = ukdf.dropna()
    kanji_level = ukdf[["kanji", "level"]].set_index("kanji").to_dict()["level"]
    return kanji_level


def sanity_check_notes(notes: List[str]):
    """Perform simple string counting to validate card correctness.

    Example input:
    ['<verb>食べ</><teform>て</>',
    '<noun>クッキ</><obj>を</><verb>食べ</><teform>て</>',
    '<verb>食べ</><teform>て</><adv>ください</>']

    """
    for i in notes:
        count_la = i.count("<")
        count_ra = i.count(">")
        count_slash = i.count("/")
        markers = get_all_in_tri_brack(i)
        for m in markers:
            assert m in ALL_POS + ["/"], "Unknown marker: {}".format(m)
        assert (count_la + count_ra) % 2 == 0
        assert count_la == count_ra, "Unmatched <> {}".format(i)
        assert count_la / 2 == count_slash, "Unmatched </>: {}".format(i)


def get_katakana_parts(sentence: str):
    """Extract katakana words from japanese sentence."""
    kk_inc = set(get_katakana(sentence))
    kk_words = []
    kk_word = ""
    for ch in sentence:
        if ch in kk_inc:
            kk_word += ch
        else:
            kk_words.append(kk_word)
            kk_word = ""
    kk_words = [i for i in kk_words if len(i) > 0 and i != "ー"]
    return kk_words


def get_unknown_dictform_words(sentence: str, known_kanji: set) -> dict:
    """Get dictionary form of unknown words given a sentence and set of known kanji."""
    tagger = MeCab.Tagger()
    words_dict = [i.split() for i in tagger.parse(sentence).split("\n")]
    dict_forms = {}
    for word in words_dict:
        if word[0] == "EOS":
            break
        if len(word) > 3:
            dict_forms[word[0]] = word[3]
    keep = {}
    for item in dict_forms.items():
        if not set(get_kanji(item[0])).issubset(known_kanji):
            keep[item[0]] = item[1]
    return keep
