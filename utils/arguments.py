import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='model name')
    parser.add_argument('--dataset', '-d', type=str, default='amazon', help='dataset name')
    parser.add_argument('--seed', type=int, default=2024)
    return parser