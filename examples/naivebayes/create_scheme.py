import sys

import argparse

AVG_COLS = 12
COV_COLS = 78

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest='avg_cols', type=int, required=True, help='number of timbre average columns')
    parser.add_argument("-c", dest='cov_cols', type=int, required=True, help='number of timbre covariance columns')
    parser.add_argument("--no-id", dest='id', action='store_false', default=True, help='include an id [default=true]')
    parser.add_argument("--no-y", dest='y', action='store_false', default=True, help='include an id [default=true]')
    parser.add_argument("--input", dest='inputtype', help="test or train", required=True)

    opt = parser.parse_args(sys.argv[1:])

    assert opt.avg_cols <= AVG_COLS
    assert opt.cov_cols <= COV_COLS

    sch = []
    if opt.id:
      sch.append(('id', 'LONG_TYPE',))

    if opt.y:
      sch.append(('y', 'LONG_TYPE',))

    for i in range(opt.avg_cols):
        sch.append(('x{0}'.format(i), 'DOUBLE_TYPE',))

    for i in range(opt.cov_cols):
        sch.append(('x{0}'.format(i+opt.avg_cols), 'DOUBLE_TYPE'))

    cat = {}

    if opt.inputtype == 'train':
      cat['public:adhoc:trainingdata'] = sch
    else:
      cat['public:adhoc:testdata'] = sch

    print cat
