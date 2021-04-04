"""Get wanikani vocab and build sample sentences."""
import re

import pandas as pd
import pykakasi
from tqdm import tqdm
from wanikani_api.client import Client

import jplearning as jpl

tqdm.pandas()


def get_all_in_tri_brack(s: str) -> list:
    """Extract all words inside triangular brackets in a string."""
    return re.findall("\\<(.*?)\\>", s)


def get_furigana(text, known_kanji={}):
    """Return furigana replacements."""
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    replacements = {}
    suffix_removal = ["て", "で", "く"]
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
