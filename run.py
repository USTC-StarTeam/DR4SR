import os
import yaml
import quickstart
import pprint

from utils import *

if __name__ == '__main__':
    parser = get_default_parser()
    args = vars(parser.parse_args())

    model_config = yaml.load(os.path.join('configs', args['model']))
    args.update(model_config)
    pprint.pprint(args)

    setup_environment(args)

    quickstart.run(args)
