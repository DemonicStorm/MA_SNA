import pandas as pd
import pickle
from nltk import wordpunct_tokenize
import datetime
import re
from somajo import SoMaJo

tokenizer = SoMaJo("de_CMC")


def split_names(df,columnname):
    """
    Appends splitted nameparts to df.
    :param df: DataFrame
    :param columnname: column containing fullnames
    :return: df with two additional columns (firstname, lastname)
    """
    data = df[columnname].to_list()
    fname = []
    lname = []
    for name in data:
        split = name.split(" ")
        if len(split) == 1:
            fname.append("")
            lname.append(split[0])
        elif len(split) == 2:
            fname.append(split[0])
            lname.append(split[1])
        elif len(split) == 3:
            if split[1].lower() in ["von", "van", "de", "du"]:
                fname.append(split[0])
                lname.append(" ".join(split[1:]))
            else:
                fname.append(" ".join(split[:2]))
                lname.append(split[2])
        elif len(split) == 4:
            if split[1] == "de" and split[2] == "la":
                fname.append(split[0])
                lname.append(" ".join(split[1:]))
            elif split[-2] in ["von", "van", "de", "du", "la"]:
                fname.append(" ".join(split[:-2]))
                lname.append(" ".join(split[-2:]))
            else:
                fname.append(" ".join(split[:-1]))
                lname.append(split[-1])
        elif len(split) > 4:
            if split[-2].lower() in ["von", "van", "de", "du", "la"]:
                if split[-3] == "de":
                    fname.append(" ".join(split[:-3]))
                    lname.append(" ".join(split[-3:]))
                else:
                    fname.append(" ".join(split[:-2]))
                    lname.append(" ".join(split[-2:]))
            else:
                fname.append(" ".join(split[:-1]))
                lname.append(" ".join(split[-1]))

    df["firstname"] = fname
    df["lastname"] = lname
    return df


def sentenizer(speech):
    sentenized = tokenizer.tokenize_text([speech])
    out = []
    for sentence in sentenized:
        sent = []
        for token in sentence:
            sent.append(token.text)
        sentnce = " ".join(sent[:-1])
        sent_punct = "".join([sentnce, sent[-1]])
        out.append(sent_punct)
    return out


def re_with_sentence(speeches,names):
    # Kompiliere die Muster einmal vor der Schleife
    namejoinf = "\\b|\\b".join([x for x in names["phil"]])
    namejoinl = "\\b|\\b".join([x for x in names["lastname"]])
    fpattern = re.compile("".join(["\\b", namejoinf, "\\b"]))
    lpattern = re.compile("".join(["\\b", namejoinl, "\\b"]))
    sentend = re.compile(r'\. ')
    out = []
    # Verwende itertuples für effizientere Iteration
    for row in speeches.itertuples(index=False):
        if not isinstance(row.speechContent, str):
            continue

        full_mentions_iter = list(fpattern.finditer(row.speechContent))
        mentions_iter = list(lpattern.finditer(row.speechContent))

        full_mentions_indices = [m.span() for m in full_mentions_iter]
        mentions_indices = [m.span() for m in mentions_iter]

        full_mentions = [m.group() for m in full_mentions_iter]
        mentions = [m.group() for m in mentions_iter]

        if full_mentions or mentions != []:  # Überprüfe direkt auf nicht-leere Listen
            sentences_full = []
            sentences = []
            sentenized = sentenizer(row.speechContent)
            for s in sentenized:
                full = fpattern.findall(s)
                last = lpattern.findall(s)
                if len(full) > 0:
                    for x in range(len(full)):
                        sentences_full.append(s)
                if len(last) > 0:
                    for x in range(len(last)):
                        sentences.append(s)

            out.append({
                "id": row.id,
                "lastName": row.lastName,
                "politicianID": row.politicianId,
                "factionID": row.factionId,
                "date": row.date,
                "fullMentions": full_mentions,
                "mentions": mentions,
                "fm_sent": sentences_full,
                "m_sent": sentences
            })
    return out

def filter_irrelevant(df):
    exclude_values = ['präsident', 'schriftführer', 'vizepräsident']
    return df[~df["positionLong"].isin(exclude_values)]

def check_list_lengths(df):
    inconsistent_rows = []
    for i, row in df.iterrows():
        # Finde die Länge aller Spalten, die Listen enthalten
        list_lengths = [len(val) for val in row if isinstance(val, list)]
        # Überprüfen, ob alle Längen übereinstimmen
        if len(set(list_lengths)) > 1:
            inconsistent_rows.append(i)
    return inconsistent_rows




if __name__ == "__main__":
    print("imported")


    #  workflow
    speeches = pd.read_csv("speeches.csv",encoding="UTF-8")
    names = pd.read_csv("PHIL_.csv",sep=";",encoding="UTF-8")
    reduced = filter_irrelevant(speeches)
    results = re_with_sentence(reduced,names)

    import sentiws
    results_df = sentiws.sentis(results)
    datetime.datetime.now()
    results_df.to_csv("_".join(["both_results",datetime.datetime.now().strftime("%Y%m%d_%H%M"),".csv"]),index=False,sep=";")
    fullMentions = results_df.drop(["mentions","m_sent","m_senti"],axis=1)
    mentions = results_df.drop(["fullMentions","fm_sent","fm_senti"],axis=1)

    # split data in fullMentions & Mentions
    inconsistent_rows_fM = check_list_lengths(fullMentions)
    inconsistent_rows_M = check_list_lengths(mentions)

    # drop inconsistent rows due to reference without sentiment, to unnest Dataset
    fM_clean = fullMentions.drop(index=inconsistent_rows_fM)
    m_clean = mentions.drop(index=inconsistent_rows_M)
    unnested_fM = fM_clean.explode(["fullMentions","fm_sent","fm_senti"]).reset_index(drop=True)
    unnested_fM = unnested_fM.dropna(subset="fullMentions")
    unnested_M = m_clean.explode(["mentions","m_sent","m_senti"]).reset_index(drop=True)
    # renaming for Gephi
    unnested_fM = unnested_fM.rename(columns={"fullMentions":"target"})
    unnested_M = unnested_M.rename(columns={"mentions":"target"})

    #  dekodierung der Parteinamen
    with open("factions.pkl", "rb") as f:
            factions = pickle.load(f)
    fact_dict = factions.to_dict()
    f_d = {fact_dict["id"][x]:fact_dict["abbreviation"][x] for x in list(fact_dict["id"])}

    unnested_fM["source"] = unnested_fM["factionID"].map(f_d)
    unnested_M["source"] = unnested_M["factionID"].map(f_d)
    unnested_fM = unnested_fM.drop("factionID",axis=1)
    unnested_M = unnested_M.drop("factionID",axis=1)
    unnested_fM = unnested_fM.rename(columns={"fullMentions":"target","id":"redeID"})
    unnested_M = unnested_M.rename(columns={"mentions":"target","id":"redeID"})

    schools_full = {x.phil:x.schule for x in names.itertuples()}
    schools_short = {x.lastname:x.schule for x in names.itertuples()}
    unnested_fM["school"] = unnested_fM["target"].map(schools_full)
    unnested_M["school"] = unnested_M["target"].map(schools_short)
    school_color_dict = {x.schule:x.color for x in names.itertuples()}
    unnested_fM["color"] = unnested_fM["school"].map(school_color_dict)
    unnested_M["color"] = unnested_M["school"].map(school_color_dict)
    unnested_fM["id"] = range(len(unnested_fM))
    unnested_M["id"] = range(len(unnested_M))
    # numerische kodierung der Sentiments
    unnested_fM["fm_senti"] = sentiws.num_weight(unnested_fM,"fm_senti")
    unnested_M["m_senti"] = sentiws.num_weight(unnested_M,"m_senti")
    unnested_fM.to_csv("_".join(["results_fM",datetime.datetime.now().strftime("%Y%m%d_%H%M"),".csv"]),sep=";",index=False)
    unnested_M.to_csv("_".join(["results_M",datetime.datetime.now().strftime("%Y%m%d_%H%M"),".csv"]),sep=";",index=False)

