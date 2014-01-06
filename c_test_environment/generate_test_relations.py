import random

def generate(basename, fields, tuples, datarange):
    with open(basename+str(fields), 'w') as f:
        for i in range(0,tuples):
            for j in range(0,fields):
                dat = random.randint(0, datarange)
                f.write(str(dat))
                f.write(' ')
            f.write("\n")

if __name__ == "__main__":
    print 'generating'
    for n in ['R','S','T']:
        for nf in [1,2,3]:
            generate(n, nf, 30, 10)
