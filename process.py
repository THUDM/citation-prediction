import os
import json
import pandas as pd
import csv
import numpy as np
from collections import defaultdict as dd
from tqdm import tqdm


def gen_test_author_per_year_citation_30(pred_year=2016):
    author_to_year_to_citation = dd(lambda: dd(int))
    if pred_year == 2016:
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
    elif pred_year == 2022:
        paper_dict = {}
        print("loading dblp papers...")
        with open("data/2022/raw/dblp_papers_before_2017.json") as rf:
            papers = json.load(rf)
        print("dblp papers loaded")

        for paper in tqdm(papers):
            paper_dict[paper["_id"]] = paper

        aids_test = []
        with open("data/2022/processed/authors_test.json") as rf:
            for line in rf:
                aids_test.append(json.loads(line)["id"])
        aids_set = set(aids_test)
        
        for paper in tqdm(papers):
            cur_refs = paper.get("references", [])
            if len(cur_refs) == 0:
                continue
            cur_year = paper["year"]
            for ref_id in cur_refs:
                ref_paper = paper_dict.get(ref_id, None)
                if ref_paper is None:
                    continue
                ref_authors = ref_paper.get("authors", [])
                for a in ref_authors:
                    cur_aid = a.get("_id", None)
                    if cur_aid is None:
                        continue
                    if cur_aid in aids_set:
                        author_to_year_to_citation[cur_aid][cur_year] += 1
        with open("data/2022/processed/cnt_test_30_new.csv", "w") as wf:
            wf.write("author_id,total_citation," + ",".join([str(i) for i in range(1987, 2017)]) + "\n")
            for author in tqdm(aids_test):
                cur_total_citation = sum(author_to_year_to_citation[author].values())
                wf.write(author + "," + str(cur_total_citation) + "," + ",".join([str(author_to_year_to_citation[author].get(i, 0)) for i in range(1987, 2017)]) + "\n")

    else:
        raise ValueError("pred_year should be 2016 or 2022")


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


def feature_extraction_seq(pred_year=2016):
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
    n_pubs_by_year = dd(dict)
    paper_to_year = {}

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
            paper_to_year[paper] = year 
            for a in authors:
                if year in n_pubs_by_year.get(a, {}):
                    n_pubs_by_year[a][year].add(paper)
                else:
                    n_pubs_by_year[a][year] = set([paper])
            if year > 2006:
                if meeting not in meeting_to_paper:
                    meeting_to_paper[meeting] = {}
                meeting_to_paper[meeting][paper] = 0
    print("Finish processing paper_to_author[] & total_papers[] ")
    # print(n_pubs_by_year)
    for a in tqdm(n_pubs_by_year):
        print(a, n_pubs_by_year[a])

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

    train_feature_seq = []
    y_train = []
    for line in data_citation_train:
        author, result = line[1], line[2]
        cur_feature = np.zeros((20, 2))
        for i in range(1992, 2012):
            cur_feature[i - 1992][0] = citation_by_year[author][i]
            cur_feature[i - 1992][1] = len(n_pubs_by_year[author].get(i, set()))
        train_feature_seq.append(cur_feature)
        y_train.append(int(result))
    train_feature_seq = np.array(train_feature_seq)
    y_train = np.array(y_train)

    test_feature_seq = []
    y_test = []
    for i, line in enumerate(data_citation_test):
        author = line[1]
        result = data_citation_result[i][1]
        cur_feature = np.zeros((20, 2))
        for i in range(1992, 2012):
            cur_feature[i - 1992][0] = citation_by_year[author][i]
            cur_feature[i - 1992][1] = len(n_pubs_by_year[author].get(i, set()))
        test_feature_seq.append(cur_feature)
        y_test.append(int(result))
    
    test_feature_seq = np.array(test_feature_seq)
    y_test = np.array(y_test)

    np.save("data/{}/processed/train_feature_seq.npy".format(pred_year), train_feature_seq)
    np.save("data/{}/processed/test_feature_seq.npy".format(pred_year), test_feature_seq)
    np.save("data/{}/processed/y_train.npy".format(pred_year), y_train)
    np.save("data/{}/processed/y_test.npy".format(pred_year), y_test)


def gen_dynamic_graph_data(pred_year=2016):
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

    per_year_graphs = [dd(dict) for _ in range(1992, 2012)]

    citation_by_year = {}  # citation_by_year[author][year] = ?
    citation_by_paper = {}  # citation_by_paper[author][paper] = ?
    total_citation = {}  # total_citation[author] = ?
    total_papers = {}  # total_papers[author] = ?
    h_index = {}  # h_index[author] = ?

    paper_to_author = {}  # paper_to_author[paper] = {}
    meeting_to_paper = {}  # meeting_to_paper[meeting][paper] = ?
    n_pubs_by_year = dd(dict)

    for line in data_author:
        citation_by_year[line[1]] = {}
        for i in range(1992, 2012):
            citation_by_year[line[1]][i] = 0
        citation_by_paper[line[1]] = {}
        total_papers[line[1]] = 0
        total_citation[line[1]] = 0
        h_index[line[1]] = 0
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
            for a in authors:
                if year in n_pubs_by_year.get(a, {}):
                    n_pubs_by_year[a][year].add(paper)
                else:
                    n_pubs_by_year[a][year] = set([paper])
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

    a_idx_max = 0

    name_to_idx = {}
    for i, line in enumerate(data_citation_test):
        author = line[1]
        idx = int(line[0])
        name_to_idx[author] = idx
        a_idx_max = max(a_idx_max, idx)
    
    for i, line in enumerate(data_citation_train):
        author = line[1]
        idx = int(line[0])
        name_to_idx[author] = idx  
        a_idx_max = max(a_idx_max, idx) 

    year = 0
    for line in tqdm(data_paper):
        if line[0][0:2] == "#t":
            year = int(line[0][2:])
        if line[0][0:2] == "#%":
            paper = line[0][2:]
            if paper in paper_to_author:
                cur_authors = paper_to_author[paper]
                # for author in paper_to_author[paper]:
                if year < 1992:
                    continue
                for i in range(len(cur_authors)):
                    if cur_authors[i] not in name_to_idx:
                        name_to_idx[cur_authors[i]] = a_idx_max + 1
                        a_idx_max += 1
                    for j in range(len(cur_authors)):
                        if i != j:
                            if cur_authors[j] not in per_year_graphs[year - 1992].get(cur_authors[i], {}):
                                per_year_graphs[year - 1992][cur_authors[i]][cur_authors[j]] = 1
                            else:
                                per_year_graphs[year - 1992][cur_authors[i]][cur_authors[j]] += 1
      
    
    with open("data/{}/processed/dynamic_coauthor_graph.txt".format(pred_year), 'w') as f:
        f.write("author_id\tcoauthor_id\tweight\tyear\n")
        for year in range(1992, 2012):
            for author in per_year_graphs[year - 1992]:
                for coauthor in per_year_graphs[year - 1992][author]:
                    f.write("{}\t{}\t{}\t{}\n".format(name_to_idx[author], name_to_idx[coauthor], per_year_graphs[year - 1992][author][coauthor], year))
                    f.flush()
    
    idx_to_name = {}
    for name in name_to_idx:
        idx_to_name[name_to_idx[name]] = name

    with open("data/{}/processed/idx_to_name.txt".format(pred_year), 'w') as f:
        for idx in idx_to_name:
            f.write("{}\t{}\n".format(idx, idx_to_name[idx]))
            f.flush()
    
    author_features = np.zeros((a_idx_max + 1, 20, 2))
    for author in name_to_idx:
        for i in range(1992, 2012):
            author_features[name_to_idx[author]][i-1992][0] = citation_by_year[author].get(i, 0)
            author_features[name_to_idx[author]][i-1992][1] = len(n_pubs_by_year[author].get(i, set()))

    np.save("data/{}/processed/author_features_all.npy".format(pred_year), author_features)


def split_data_author_inf_2022():
    with open("data/2022/raw/gs_citation_2022_result.json") as rf:
        all_gs_authors = []
        for line in tqdm(rf):
            cur_author = json.loads(line)
            all_gs_authors.append(cur_author)

    np.random.seed(2022)
    np.random.shuffle(all_gs_authors)
    n_authors = len(all_gs_authors)
    authors_train = all_gs_authors[: int(n_authors * 0.6)]
    authors_valid = all_gs_authors[int(n_authors * 0.6): int(n_authors * 0.8)]
    authors_test = all_gs_authors[int(n_authors * 0.8):]

    out_dir = "data/2022/processed/"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + "authors_train.json", 'w') as f:
        for author in authors_train:
            f.write(json.dumps(author) + "\n")
    
    with open(out_dir + "authors_valid.json", 'w') as f:
        for author in authors_valid:
            f.write(json.dumps(author) + "\n")
    
    with open(out_dir + "authors_test.json", 'w') as f:
        for author in authors_test:
            f.write(json.dumps(author) + "\n")


def feature_extraction_2022():
    print("loading dblp papers...")
    with open("data/2022/raw/dblp_papers_before_2017.json") as rf:
        papers = json.load(rf)
    print("dblp papers loaded") 

    citation_by_year = dd(lambda: dd(int))  # citation_by_year[author][year] = ?
    citation_by_paper = dd(lambda: dd(int))  # citation_by_paper[author][paper] = ?
    total_citation = dd(int)  # total_citation[author] = ?
    total_papers = dd(int)  # total_papers[author] = ?
    h_index = dd(int)  # h_index[author] = ?

    paper_to_author = {}  # paper_to_author[paper] = {}
    meeting_to_paper = dd(lambda: dd(int))  # meeting_to_paper[meeting][paper] = ?
    h_rank = {}  # h_rank[meeting] = ?
    p_rank = {}
    author_rank1 = dd(int)
    author_rank2 = dd(int)
    paper_dict = {}

    for paper in tqdm(papers):
        cur_authors = paper.get("authors", [])
        cur_aids = []
        for author in cur_authors:
            aid = author.get("_id")
            if aid is None:
                continue
            total_papers[aid] += 1
            cur_aids.append(aid)
        year = paper.get("year", 0)
        meeting = paper.get("venue", {}).get("raw", "")
        pid = paper["_id"]
        paper_to_author[pid] = cur_aids
        meeting_to_paper[meeting][pid] = 0
        paper_dict[pid] = paper

    print("Finish processing paper_to_author[] & total_papers[] ")

    for paper in tqdm(papers):
        cur_refs = paper.get("references", [])
        year = paper.get("year", 0)
        for ref in cur_refs:
            ref_paper = paper_dict.get(ref, {})
            cur_authors = ref_paper.get("authors", [])
            for author in cur_authors:
                aid = author.get("_id")
                if aid is None:
                    continue
                if year >= 1997:
                    citation_by_year[aid][year] += 1
                total_citation[aid] += 1
                citation_by_paper[aid][ref] += 1
    print("Finish processing citation_by_year[] & total_citation[] & citation_by_paper[]")

    true_citation = {}
    rank_by_paper = {}
    with open("data/2022/processed/authors_train.json") as rf:
        for line in tqdm(rf):
            cur_author = json.loads(line)
            aid = cur_author["id"]
            true_citation[aid] = cur_author["Different"]
    
    with open("data/2022/processed/authors_valid.json") as rf:
        for line in tqdm(rf):
            cur_author = json.loads(line)
            aid = cur_author["id"]
            true_citation[aid] = cur_author["Different"]

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
                # print("author", author, rank_by_paper[author])
    print("Finish processing rank_by_paper")

    true_citation_test = {}
    with open("data/2022/processed/authors_test.json") as rf:
        for line in tqdm(rf):
            cur_author = json.loads(line)
            aid = cur_author["id"]
            true_citation_test[aid] = cur_author["Different"]

    aids_used = set(true_citation.keys()) | set(true_citation_test.keys())
    for aid in tqdm(aids_used):
        contain = rank_by_paper.get(aid)
        try:
            print("-----------------------", aid, contain, int(np.sum(np.array(list(contain.values())))))
        except:
            pass
        if contain is None:
            author_rank1[aid] = 50
        else:
            author_rank1[aid] = int(np.sum(np.array(list(contain.values()))))
        print("author_rank1", author_rank1[aid])
        papers = sorted(citation_by_paper[aid].items(), key=lambda x: x[1], reverse=True)
        for i in range(len(papers)):
            if i + 1 > papers[i][1]:
                h_index[aid] = i
                break
    print("Finish processing h_index[] & author_rank1")

    # for aid in tqdm(true_citation):
    #     print("rank_by_paper train", aid, author_rank1[aid], rank_by_paper[aid])
    
    # for aid in tqdm(true_citation_test):
    #     print("rank_by_paper test", aid, author_rank1[aid], rank_by_paper.get(aid))

    for meeting in meeting_to_paper:
        for paper in meeting_to_paper[meeting]:
            if len(paper_to_author[paper]) == 0:
                continue
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

    with open("data/2022/processed/train.csv", "w") as wf:
        wf.write("id,author_rank1,author_rank2,total_citation,total_papers,h_index," +
        ",".join([str(x) for x in range(1997, 2017)]) + ",result\n")
        for aid in tqdm(true_citation):
            wf.write(str(aid) + "," + str(author_rank1[aid]) + "," + str(author_rank2[aid]) + "," + str(total_citation[aid]) + "," + 
            str(total_papers[aid]) + "," + str(h_index[aid]) + "," + ",".join([str(citation_by_year[aid][x]) for x in range(1997, 2017)]) + 
            "," + str(true_citation[aid]) + "\n")
    print("Finish writing train.csv")
    

    with open("data/2022/processed/test.csv", "w") as wf:
        wf.write("id,author_rank1,author_rank2,total_citation,total_papers,h_index," +
        ",".join([str(x) for x in range(1997, 2017)]) + ",result\n")
        for aid in tqdm(true_citation_test):
            wf.write(str(aid) + "," + str(author_rank1[aid]) + "," + str(author_rank2[aid]) + "," + str(total_citation[aid]) + "," + 
            str(total_papers[aid]) + "," + str(h_index[aid]) + "," + ",".join([str(citation_by_year[aid][x]) for x in range(1997, 2017)]) + 
            "," + str(true_citation_test[aid]) + "\n")
    print("Finish writing test.csv")


def feature_extraction_seq_2022():
    print("loading dblp papers...")
    with open("data/2022/raw/dblp_papers_before_2017.json") as rf:
        papers = json.load(rf)
    print("dblp papers loaded") 

    citation_by_year = dd(lambda: dd(int))  # citation_by_year[author][year] = ?
    citation_by_paper = dd(lambda: dd(int))  # citation_by_paper[author][paper] = ?
    total_citation = dd(int)  # total_citation[author] = ?
    total_papers = dd(int)  # total_papers[author] = ?
    h_index = dd(int)  # h_index[author] = ?

    paper_to_author = {}  # paper_to_author[paper] = {}
    meeting_to_paper = dd(lambda: dd(int))  # meeting_to_paper[meeting][paper] = ?
    h_rank = {}  # h_rank[meeting] = ?
    p_rank = {}
    author_rank1 = dd(int)
    author_rank2 = dd(int)
    paper_dict = {}    
    n_pubs_by_year = dd(lambda: dd(set))
    paper_to_year = {}

    for paper in tqdm(papers):
        pid = paper.get["_id"]
        year = paper.get("year", 0)
        aids = []
        for author in paper.get("authors", []):
            aid = author.get("_id")
            if aid is None:
                continue
            total_papers[aid] += 1
            aids.append(aid)
        paper_to_author[pid] = aids
        paper_to_year[pid] = year
        for a in aids:
            n_pubs_by_year[a][year].add(pid)
        paper_dict[pid] = paper
    print("Finish processing paper_to_author[] & total_papers[] ")
    # print(n_pubs_by_year)
    for a in tqdm(n_pubs_by_year):
        print(a, n_pubs_by_year[a])
    
    for pid in tqdm(paper_dict):
        paper = paper_dict[pid]
        year = paper.get("year", 0)
        if year == 0:
            continue
        for ref_id in paper.get("references", []):
            if ref_id not in paper_to_author:
                continue
            for aid in paper_to_author[ref_id]:
                citation_by_paper[aid][ref_id] += 1
                citation_by_year[aid][year] += 1
                total_citation[aid] += 1
    print("Finish processing citation_by_year[] & citation_by_paper[]")

    train_feature_seq = []
    y_train = []
    with open("data/2022/processed/authors_train.json") as rf:
        for line in tqdm(rf):
            cur_author = json.loads(line)
            aid = cur_author["id"]
            y_train.append(cur_author["Different"])
            cur_feature = np.zeros((20, 2))
            for i in range(1997, 2017):
                cur_feature[i - 1997][0] = citation_by_year[aid][i]
                cur_feature[i - 1997][1] = len(n_pubs_by_year[aid][i])
            train_feature_seq.append(cur_feature)
            if len(train_feature_seq) % 1000 == 0:
                print(cur_feature)
    
    with open("data/2022/processed/authors_valid.json") as rf:
        for line in tqdm(rf):
            cur_author = json.loads(line)
            aid = cur_author["id"]
            y_train.append(cur_author["Different"])
            cur_feature = np.zeros((20, 2))
            for i in range(1997, 2017):
                cur_feature[i - 1997][0] = citation_by_year[aid][i]
                cur_feature[i - 1997][1] = len(n_pubs_by_year[aid][i])
            train_feature_seq.append(cur_feature)
            if len(train_feature_seq) % 1000 == 0:
                print(cur_feature)

    train_feature_seq = np.array(train_feature_seq)
    y_train = np.array(y_train)

    test_feature_seq = []
    y_test = []
    with open("data/2022/processed/authors_test.json") as rf:
        for line in tqdm(rf):
            cur_author = json.loads(line)
            aid = cur_author["id"]
            y_test.append(cur_author["Different"])
            cur_feature = np.zeros((20, 2))
            for i in range(1997, 2017):
                cur_feature[i - 1997][0] = citation_by_year[aid][i]
                cur_feature[i - 1997][1] = len(n_pubs_by_year[aid][i])
            test_feature_seq.append(cur_feature)
            if len(test_feature_seq) % 1000 == 0:
                print(cur_feature)
    
    test_feature_seq = np.array(test_feature_seq)
    y_test = np.array(y_test)

    pred_year = 2022
    np.save("data/{}/processed/train_feature_seq.npy".format(pred_year), train_feature_seq)
    np.save("data/{}/processed/test_feature_seq.npy".format(pred_year), test_feature_seq)
    np.save("data/{}/processed/y_train.npy".format(pred_year), y_train)
    np.save("data/{}/processed/y_test.npy".format(pred_year), y_test)


if __name__ == "__main__":
    pred_year = 2022
    # gen_test_author_per_year_citation_30(pred_year=2022)
    if pred_year == 2016:
        # feature_extraction(pred_year=pred_year)
        # feature_extraction_seq(pred_year=2016)
        pass
    elif pred_year == 2022:
        # feature_extraction_2022()
        feature_extraction_seq_2022()
        pass
    else:
        raise NotImplementedError
    
    # gen_dynamic_graph_data(pred_year=2016)
    # split_data_author_inf_2022()
