from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import utils as u

class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''
    def __init__(self,args,tasker):
        
        
        if tasker.is_static: #### For static datsets
            assert args.train_proportion + args.dev_proportion < 1, \
                'there\'s no space for test samples'
            #only the training one requires special handling on start, the others are fine with the split IDX.
            
            random_perm=False
            indexes = tasker.data.nodes_with_label
            
            if random_perm:
                perm_idx = torch.randperm(indexes.size(0))
                perm_idx = indexes[perm_idx]
            else:
                print ('tasker.data.nodes',indexes.size())
                perm_idx, _ = indexes.sort()
            #print ('perm_idx',perm_idx[:10])
            
            self.train_idx = perm_idx[:int(args.train_proportion*perm_idx.size(0))]
            self.dev_idx = perm_idx[int(args.train_proportion*perm_idx.size(0)): int((args.train_proportion+args.dev_proportion)*perm_idx.size(0))]
            self.test_idx = perm_idx[int((args.train_proportion+args.dev_proportion)*perm_idx.size(0)):]
            # print ('train,dev,test',self.train_idx.size(), self.dev_idx.size(), self.test_idx.size())
            
            train = static_data_split(tasker, self.train_idx, test = False)
            train = DataLoader(train, shuffle=True,**args.data_loading_params)
            
            dev = static_data_split(tasker, self.dev_idx, test = True)
            dev = DataLoader(dev, shuffle=False,**args.data_loading_params)
            
            test = static_data_split(tasker, self.test_idx, test = True)
            test = DataLoader(test, shuffle=False,**args.data_loading_params)
                        
            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test
            
            
        else: #### For datsets with time
            assert args.train_proportion + args.dev_proportion < 1, \
                'there\'s no space for test samples'
            #only the training one requires special handling on start, the others are fine with the split IDX.
            start = tasker.data.min_time + args.num_hist_steps #-1 + args.adj_mat_time_window
            end = args.train_proportion
            
            end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
            train = data_split(tasker, start, end, test = False)
            train = DataLoader(train,**args.data_loading_params)
    
            start = end
            end = args.dev_proportion + args.train_proportion
            end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
            if args.task == 'link_pred':
                dev = data_split(tasker, start, end, test = True, all_edges=True)
            else:
                dev = data_split(tasker, start, end, test = True)

            dev = DataLoader(dev,num_workers=args.data_loading_params['num_workers'])
            
            start = end
            
            #the +1 is because I assume that max_time exists in the dataset
            end = int(tasker.max_time) + 1
            if args.task == 'link_pred':
                test = data_split(tasker, start, end, test = True, all_edges=True)
            else:
                test = data_split(tasker, start, end, test = True)
                
            test = DataLoader(test,num_workers=args.data_loading_params['num_workers'])
            
            print ('Dataset splits sizes:  train',len(train), 'dev',len(dev), 'test',len(test))
            
            
            
            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test
        


class data_split(Dataset):
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        return self.end-self.start

    def __getitem__(self,idx):
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test = self.test, **self.kwargs)
        return t


class static_data_split(Dataset):
    def __init__(self, tasker, indexes, test):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.indexes = indexes
        self.test = test
        self.adj_matrix = tasker.adj_matrix

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self,idx):
        idx = self.indexes[idx]
        return self.tasker.get_sample(idx,test = self.test)


class Author_Inf_Splitter():

    def __init__(self, args, tasker):
        year = tasker.data.year
        self.tasker = tasker

        if year == 2016:
            train_idx = []
            with open("../data/{}/raw/citation_train.txt".format(year)) as rf:
                for i, line in enumerate(tqdm(rf)):
                    if i == 0:
                        continue
                    train_idx.append(int(line.strip().split()[0]))
            
            test_idx = []
            with open("../data/{}/raw/citation_result.txt".format(year)) as rf:
                for i, line in enumerate(tqdm(rf)):
                    if i == 0:
                        continue
                    test_idx.append(int(line.strip().split()[0]))
        elif year == 2022:
            train_idx = []
            test_idx = []
            aid_to_idx = tasker.data.name_to_idx

            with open("../data/2022/processed/authors_train.json") as rf:
                for line in tqdm(rf):
                    cur_author = json.loads(line)
                    aid = cur_author["id"]
                    train_idx.append(aid_to_idx[aid])
            
            with open("../data/2022/processed/authors_valid.json") as rf:
                for line in tqdm(rf):
                    cur_author = json.loads(line)
                    aid = cur_author["id"]
                    train_idx.append(aid_to_idx[aid])
            
            with open("../data/2022/processed/authors_test.json") as rf:
                for line in tqdm(rf):
                    cur_author = json.loads(line)
                    aid = cur_author["id"]
                    test_idx.append(aid_to_idx[aid])

        # start = tasker.data.min_time + args.num_hist_steps #-1 + args.adj_mat_time_window
        start = tasker.data.min_time
        # end = int(np.floor(tasker.data.max_time.type(torch.float)))
        end = tasker.data.max_time

        train = data_split(tasker, start, end, test = False)
        train = DataLoader(train,**args.data_loading_params)

        dev = data_split(tasker, start, end, test = True)
        dev = DataLoader(dev,num_workers=args.data_loading_params['num_workers'])

        test = data_split(tasker, start, end, test = True)
        test = DataLoader(test,num_workers=args.data_loading_params['num_workers'])

        self.train = train
        self.dev = dev
        self.test = test

        n_train = len(train_idx)
        train_idx_2 = train_idx[:int(n_train*0.8)]
        dev_idx = train_idx[int(n_train*0.8):]

        self.train_idx = train_idx_2
        self.dev_idx = dev_idx
        self.test_idx = test_idx
            