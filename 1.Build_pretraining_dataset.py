import os
import torch
import argparse
from tqdm import tqdm
from random import shuffle
from sequential.seq2pat import Seq2Pat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./dataset/amazon-toys/toy/', help='The path to the training dataset.')
    parser.add_argument('--alpha', type=int, default=5, help='The sliding window size for pre-training dataset construction.')
    parser.add_argument('--beta', type=int, default=2 , help='The threshold for pre-training dataset construction.')
    parser.add_argument('--n_jobs', type=int, default=2 , help='The job number for Seq2Pat pattern mining.')
    args = parser.parse_args()
    
    # Some constant configs
    max_seq_len = 50
    
    # Load the original dataset
    seq2pat_data_path = os.path.join(args.root_path, 'seq2pat_data.pth')
    seq2pat_data = torch.load(seq2pat_data_path)
    print(f'Original dataset loaded with size {len(seq2pat_data)}')

    seq2pat = Seq2Pat(sequences=seq2pat_data, n_jobs=args.n_jobs, max_span=args.alpha)

    print('Performing rule-based pattern-mining!')
    patterns = seq2pat.get_patterns(min_frequency=args.beta)
    patterns_value = [_[:-1] for _ in patterns]
    print(f'Rule-based patterns mined with size {len(patterns)}')

    original_train_path = os.path.join(args.root_path, 'train.pth')
    original_train = torch.load(original_train_path)
    
    seq_list_ori = [_[1][:_[3]] + [_[2][_[3] - 1]] for _ in original_train]
    
    print('Pre-processing the extracted patterns for dataset regeneration.')
    def truncate_or_pad(seq):
        cur_seq_len = len(seq)
        if cur_seq_len > max_seq_len:
            return seq[-max_seq_len:]
        else:
            return seq + [0] * (max_seq_len - cur_seq_len)

    train_set = set()

    for pattern in patterns_value:
        seq = pattern
        seq_len = len(seq)
        train_set.add(tuple(
            truncate_or_pad(seq[:-1]) + truncate_or_pad(seq[1:])
        ))

    train_list = []
    for _ in list(train_set):
        train_item_seq = _[:max_seq_len]
        target_item_seq = _[max_seq_len:]
        seq_len = sum([a != 0 for a in train_item_seq])
        train_list.append([
            0,
            train_item_seq,
            target_item_seq,
            seq_len,
            [1] * seq_len + [0] * (max_seq_len - seq_len),
            [0] * max_seq_len,
        ])

    output_path = os.path.join(args.root_path, 'patterns.pth')
    torch.save(train_list + original_train, output_path)
    
    def is_sublist(sublst, lst):
        for element in sublst:
            try:
                ind = lst.index(element)
            except ValueError:
                return False
            lst = lst[ind+1:]
        return True
    
    data_generation_pair = []
    total_cnt = 0
    for seq_ori in tqdm(seq_list_ori):
        shuffle(patterns_value)
        cnt = 0
        for pattern in patterns_value:
            if is_sublist(pattern, seq_ori):
                data_generation_pair.append([seq_ori, pattern])
                cnt += 1
            if cnt == 10:
                break
    print(f'Building sequence-pattern pair dataset with size {len(data_generation_pair)}.')
    
    output_path = os.path.join(args.root_path, 'seq-pat-pair.pth')
    torch.save(data_generation_pair, output_path)