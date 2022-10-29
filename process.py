import pandas as pd
import csv
import numpy as np
from collections import defaultdict as dd
from tqdm import tqdm


def gen_test_author_per_year_citation_30(pred_year=2016):
    author_to_year_to_citation = dd(lambda: dd(int))
    a_list = []
    a_set = set()
    with open("data/{}/raw/citation_test.txt".format(pred_year)) as rf:
        for line in tqdm(rf):
            a_idx, author_name = line.strip().split("\t")
            a_list.append(author_name)
            a_set.add(author_name)

    paper_id_to_author = dd(set)

    cur_paper_dict = {}
    with open("data/{}/raw/paper.txt".format(pred_year)) as rf:
        for line in tqdm(rf):
            if line.startswith("#*"):
                if "pid" in cur_paper_dict and "authors" in cur_paper_dict:
                    paper_id_to_author[cur_paper_dict["pid"]] = cur_paper_dict["authors"]
                cur_paper_dict = {}
            elif line.startswith("#index"):
                cur_pid = line.strip().split()[-1][6:]
                cur_paper_dict["pid"] = cur_pid
            elif line.startswith("#@"):
                cur_paper_dict["authors"] = line.strip()[2:].split(", ")
    # print(paper_id_to_author)

    # raise
    
    cur_paper_dict = {}
    with open("data/{}/raw/paper.txt".format(pred_year)) as rf:
        for line in tqdm(rf):
            if line.startswith("#*"):
                if "references" in cur_paper_dict:
                    # print("here1")
                    for ref in cur_paper_dict["references"]:
                        # print("here2")
                        if ref in paper_id_to_author:
                            # print("here3")
                            for author in paper_id_to_author[ref]:
                                # print("here4")
                                if author in a_set:
                                    # print("here5")
                                    author_to_year_to_citation[author][cur_paper_dict["year"]] += 1
                cur_paper_dict = {}
            elif line.startswith("#index"):
                cur_pid = line.strip().split()[-1][6:]
                cur_paper_dict["pid"] = cur_pid
            elif line.startswith("#t"):
                cur_paper_dict["year"] = int(line.strip()[2:])
            elif line.startswith("#%"):
                if "references" in cur_paper_dict:
                    cur_paper_dict["references"].append(line.strip()[2:])
                else:
                    cur_paper_dict["references"] = [line.strip()[2:]]

    with open("data/{}/processed/cnt_test_30_new.csv".format(pred_year), "w") as wf:
        wf.write("author_name," + ",".join([str(i) for i in range(1982, 2012)]) + "\n")
        for author in tqdm(a_list):
            wf.write(author + "," + ",".join([str(author_to_year_to_citation[author][i]) for i in range(1982, 2012)]) + "\n")


def feature_extraction(pred_year=2016):
    data_author = pd.read_csv("data/{}/raw/author.txt".format(pred_year),
                              header=None, sep='\t', encoding='utf8').values
    data_paper = pd.read_csv("data/{}/raw/paper.txt".format(pred_year), header=None,
                             sep='\t', encoding='utf8').values
    data_citation_train = pd.read_csv(
        "data/{}/raw/citation_train.txt".format(pred_year), header=None, sep='\t', encoding='utf8').values
    data_citation_test = pd.read_csv(
        "data/{}/raw/citation_test.txt".format(pred_year), header=None, sep='\t', encoding='utf8').values
    data_citation_result = pd.read_csv(
        "data/{}/raw/citation_result.txt".format(pred_year), header=None, sep='\t', encoding='utf8').values
    print("Finish loading data")

    citation_by_year = {}  # citation_by_year[author][year] = ?
    citation_by_paper = {}  # citation_by_paper[author][paper] = ?
    total_citation = {}  # total_citation[author] = ?
    total_papers = {}  # total_papers[author] = ?
    h_index = {}  # h_index[author] = ?

    paper_to_author = {}  # paper_to_author[paper] = {}
    meeting_to_paper = {}  # meeting_to_paper[meeting][paper] = ?
    h_rank = {}  # h_rank[meeting] = ?
    p_rank = {}
    author_rank1 = {}
    author_rank2 = {}

    for line in data_author:
        citation_by_year[line[1]] = {}
        for i in range(1992, 2012):
            citation_by_year[line[1]][i] = 0
        citation_by_paper[line[1]] = {}
        total_papers[line[1]] = 0
        total_citation[line[1]] = 0
        h_index[line[1]] = 0
        author_rank1[line[1]] = 0
        author_rank2[line[1]] = 0
    print("Init the features")

    authors = []
    meeting = ""
    year = 0
    for line in data_paper:
        if line[0][0:2] == "#@":
            authors = line[0][2:].split(', ')
            for author in authors:
                total_papers[author] += 1
        if line[0][0:2] == "#t":
            year = int(line[0][2:])
        if line[0][0:2] == "#c":
            meeting = line[0][2:]
        if line[0][0:6] == "#index":
            paper = line[0][6:]
            if paper not in paper_to_author:
                paper_to_author[paper] = authors
            else:
                paper_to_author[paper] = list(set(paper_to_author[paper] + authors))
            if year > 2006:
                if meeting not in meeting_to_paper:
                    meeting_to_paper[meeting] = {}
                meeting_to_paper[meeting][paper] = 0
    print("Finish processing paper_to_author[] & total_papers[] ")

    year = 0
    for line in data_paper:
        if line[0][0:2] == "#t":
            year = int(line[0][2:])
        if line[0][0:2] == "#%":
            paper = line[0][2:]
            if paper in paper_to_author:
                for author in paper_to_author[paper]:
                    if year >= 1992:
                        citation_by_year[author][year] += 1
                    total_citation[author] += 1
                    if paper not in citation_by_paper[author]:
                        citation_by_paper[author][paper] = 1
                    else:
                        citation_by_paper[author][paper] += 1
    print("Finish processing citation_by_year[] & citation_by_paper[]")

    true_citation = {}
    rank_by_paper = {}
    for line in data_citation_train:
        true_citation[line[1]] = line[2]
    for paper in paper_to_author:
        authors = [author for author in paper_to_author[paper] if author in true_citation]
        counts = [true_citation[author] for author in authors]
        if len(authors) > 0:
            p_rank[paper] = np.min(np.array(counts))
            index = np.where(np.array(counts) == p_rank[paper])
            auth = authors[index[0][0]]
            for x in index[0]:
                if total_citation[authors[x]] != 0:
                    auth = authors[x]
                    break
            if total_citation[auth] == 0:
                p_rank[paper] = int(p_rank[paper] / total_papers[auth])
            elif paper in citation_by_paper[auth]:
                p_rank[paper] = int(citation_by_paper[auth][paper] * p_rank[paper] / total_citation[auth])
            else:
                p_rank[paper] = 0
            for author in paper_to_author[paper]:
                if author not in rank_by_paper:
                    rank_by_paper[author] = {}
                rank_by_paper[author][paper] = p_rank[paper]
    print("Finish processing rank_by_paper")

    for line in data_author:
        author = line[1]
        contain = rank_by_paper.get(author)
        if contain is None:
            author_rank1[author] = 50
        else:
            author_rank1[author] = int(np.sum(np.array(list(contain.values()))))
        papers = sorted(citation_by_paper[author].items(), key=lambda x: x[1], reverse=True)
        for i in range(len(papers)):
            if i + 1 > papers[i][1]:
                h_index[author] = i
                break
    print("Finish processing h_index[] & author_rank1")

    for meeting in meeting_to_paper:
        for paper in meeting_to_paper[meeting]:
            tmp = paper_to_author[paper][0]
            if paper in citation_by_paper[tmp]:
                meeting_to_paper[meeting][paper] = citation_by_paper[tmp][paper]
        papers = sorted(meeting_to_paper[meeting].items(), key=lambda x: x[1], reverse=True)
        h_rank[meeting] = 0
        for i in range(len(papers)):
            if i + 1 > papers[i][1]:
                h_rank[meeting] = i
                break
        for paper in meeting_to_paper[meeting]:
            for author in paper_to_author[paper]:
                author_rank2[author] += h_rank[meeting]
    print("Finish processing author_rank2")

    with open('data/{}/processed/train.csv'.format(pred_year), 'w', newline='') as ft:
        writer = csv.writer(ft)
        writer.writerow(['author', 'author_rank1', 'author_rank2', 'total_citation', 'total_papers', 'h_index',
                         '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                             '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011',
                             'result'])
        for line in data_citation_train:
            author, result = line[1], line[2]
            data = [author, author_rank1[author], author_rank2[author], total_citation[author], total_papers[author], h_index[author]]
            for i in range(1992, 2012):
                data.append(citation_by_year[author][i])
            data.append(int(result))
            writer.writerow(data)
    print("Finish writing train.csv")

    with open('data/{}/processed/test.csv'.format(pred_year), 'w', newline='') as ft:
        writer = csv.writer(ft)
        writer.writerow(['author', 'author_rank1', 'author_rank2', 'total_citation', 'total_papers', 'h_index',
                         '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                             '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011',
                             'result'])
        for i in range(len(data_citation_test)):
            author = data_citation_test[i][1]
            result = data_citation_result[i][1]
            data = [author, author_rank1[author], author_rank2[author], total_citation[author], total_papers[author], h_index[author]]
            for j in range(1992, 2012):
                data.append(citation_by_year[author][j])
            data.append(int(result))
            writer.writerow(data)
    print("Finish writing test.csv")


if __name__ == "__main__":
    # gen_test_author_per_year_citation_30(pred_year=2016)
    feature_extraction(pred_year=2016)
