import utils as u
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch


class Author_Citation_Dataset():
    def __init__(self, args):
        year = int(args.data.split("_")[-1])
        self.year = year
        print("year: {}".format(year))
        self.feats_per_node = 2
        self.time_interval = 20

        self.name_to_idx = {}
        self.num_nodes = self.get_node_num()
        print("num_nodes: {}".format(self.num_nodes))
        self.nodes_labels = self.load_node_labels()
        self.edges = self.load_edges()
        self.nodes_feats = self.load_node_feats()
    
    def get_node_num(self):
        idx_max = 0
        with open("../data/{}/processed/idx_to_name.txt".format(self.year)) as rf:
            for i, line in enumerate(rf):
                idx = int(line.strip().split("\t")[0])
                name = line.strip().split("\t")[1]
                self.name_to_idx[name] = idx
                if idx > idx_max:
                    idx_max = idx
        idx_max += 1
        return idx_max

    def load_node_labels(self):
        labels = np.zeros((self.num_nodes, 1))

        if self.year == 2016:
            data_citation_train = pd.read_csv(
                "../data/{}/raw/citation_train.txt".format(self.year), header=None, sep='\t', encoding='utf8').values
            data_citation_result = pd.read_csv(
                "../data/{}/raw/citation_result.txt".format(self.year), header=None, sep='\t', encoding='utf8').values
            print("Finish loading labels")

            for i in range(len(data_citation_train)):
                labels[int(data_citation_train[i, 0])] = float(data_citation_train[i, -1])
            
            for i in range(len(data_citation_result)):
                labels[int(data_citation_result[i, 0])] = float(data_citation_result[i, -1])
        elif self.year == 2022:
            with open("../data/2022/raw/gs_citation_2022_result.json") as rf:
                for line in tqdm(rf):
                    cur_author = json.loads(line)
                    cur_aid = cur_author["id"]
                    labels[self.name_to_idx[cur_aid]] = cur_author["Different"]
        print("labels generated")
        
        return labels

    def load_edges(self):
        if self.year == 2016:
            delta = 1992
        elif self.year == 2022:
            delta = 1997
        else:
            raise NotImplementedError
        edges = []
        with open("../data/{}/processed/dynamic_coauthor_graph.txt".format(self.year)) as rf:
            for i, line in enumerate(tqdm(rf)):
                if i == 0:
                    continue
                line = line.strip().split("\t")
                source = int(line[0])
                target = int(line[1])
                time = int(line[-1]) - delta
                edges.append([source, target, time])
        edges = np.array(edges)
        data = torch.LongTensor(edges)

        self.max_time = int(data[:, -1].max())
        self.min_time = int(data[:, -1].min())

        data = torch.cat([data,data[:,[1,0,2]]])
        return {'idx': data, 'vals': torch.ones(data.size(0))}

    def load_node_feats(self):
        feats = np.load("../data/{}/processed/author_features_all.npy".format(self.year))  # n_nodes x time_span x dim
        print("Finish loading node features")
        return feats
