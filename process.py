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


if __name__ == "__main__":
    gen_test_author_per_year_citation_30(pred_year=2016)
