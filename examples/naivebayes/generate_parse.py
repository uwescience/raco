import sys

nfeat = int(sys.argv[1])

y = int(sys.argv[2])

if y==1:
  parse_template = "input_sp{i} = select INT(input.x{i}/{bwidth}) as value, {i} as index, y from input;"
else:
  parse_template = "input_sp{i} = select id, INT(input.x{i}/{bwidth}) as value, {i} as index from input;"

if y==1:
  print "input = SCAN(trainingdata);"
else:
  print "input = SCAN(testdata);"

bwidth = 10
for i in range(nfeat):
  print parse_template.format(i=i, bwidth=bwidth)

inputs = []
for i in range(nfeat):
  inputs.append("input_sp%d" % i)
print "input_sp = UNIONALL(%s);" % ', '.join(inputs)
