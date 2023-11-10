import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='model name')
    parser.add_argument('--dataset', '-d', type=str, default='amazon', help='dataset name')
    return parser