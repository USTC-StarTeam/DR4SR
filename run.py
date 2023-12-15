import os
import yaml
import quickstart
import pprint

from utils import *

if __name__ == '__main__':
    parser = get_default_parser()
    config = vars(parser.parse_args())

    config = load_config(config)
    pprint.pprint(config)

    setup_environment(config['train'])

    quickstart.run(config)
