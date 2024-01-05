import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.reparam_module import ReparamModule

data = torch.load('./pattern-book.pth')
print(len(data))

# Example to show how to find frequent sequential patterns
# from a given sequence database subject to constraints
from sequential.seq2pat import Seq2Pat, Attribute

# Seq2Pat over 3 sequences
seq2pat = Seq2Pat(sequences=data, n_jobs=6)

# Patterns that occur at least twice (A-D)
patterns = seq2pat.get_patterns(min_frequency=10)
patterns_value = [_[:-1] for _ in patterns]
patterns_count = [_[-1] for _ in patterns]
print(len(patterns))

original_train = torch.load('./dataset/amazon-book-seq/book/train_ori.pth')
len(original_train)

max_seq_len = 20
def truncate_or_pad(seq):
    cur_seq_len = len(seq)
    if cur_seq_len > max_seq_len:
        return seq[-max_seq_len:]
    else:
        return seq + [0] * (max_seq_len - cur_seq_len)


train_set = set()
for pattern in patterns:
    seq = pattern[:-1]
    seq_len = len(seq)
    for _ in range(1, seq_len):
        train_set.add(tuple(
            truncate_or_pad(seq[:_]) + [seq[_]],
        ))

train_list = []
for _ in train_set:
    train_list.append([
        0,
        list(_)[:-1],
        list(_)[-1],
        sum([a != 0 for a in list(_)[:-1]]),
        1,
        0
    ])

print(len(train_list))

torch.save(train_list + original_train, 'train_new.pth')