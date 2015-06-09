import csv
import sys

#TODO take a schema as input


class WordIndexer:
    def __init__(self, indexf):
        self.words = {}
        self.count = 0
        self.indexfw = open(indexf, 'w')

    def add_word(self, w):
        if w in self.words:
            return self.words[w]
        else:
            self.indexfw.write(w+'\n')
            t = self.count
            self.count += 1
            self.words[w] = t
            return t

    def close(self):
        self.indexfw.close()


def indexing(inputf):
    intfile = inputf + '.i'
    indexf = inputf + '.index'
    delim_in = ','
    delim_out = ' '

    wi = WordIndexer(indexf)
    with open(inputf, 'r') as ins:
        reader = csv.reader(ins, delimiter=delim_in)
        with open(intfile, 'w') as outs:
            writer = csv.writer(outs, delimiter=delim_out)
            for row in reader:
                cols = [wi.add_word(w) for w in row]
                writer.writerow(cols)

    wi.close()
    return intfile, indexf


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("usage: %s inputfile" % sys.argv[0])

    indexing(sys.argv[1])



