import csv

with open('sql.ans', 'r') as f1:
    with open('code.ans', 'r') as f2:
        f1csv = csv.reader(f1, delimiter=',')
        f2csv = csv.reader(f2, delimiter=',')

        f1results = {}

        for row in f1csv:
            t = tuple(row)    
            v = 0
            if t in f1results:
                v = f1results[t]

            f1results[t] = v+1

        for row in f2csv:
            t = tuple(row)
            
            if t in f1results:
                if f1results[t]==1:
                    del f1results[t]
                else:
                    v = f1results[t]
                    f1results[t] = v-1
            else:
                print t,"is not in sql"
                assert False

        for t in f1results:
            print t,"is not in code"
            assert False

print "success!"

            


