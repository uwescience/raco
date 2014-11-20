import sys

nfeat = int(sys.argv[1])

y = int(sys.argv[2])

if y==1:
  parse_template = "input_sp{i} = select INT(input.x{i}/{bwidth}) as value, {i} as index, y from input;"
else:
  parse_template = "input_sp{i} = select id, INT(input.x{i}/{bwidth}) as value, {i} as index from input;"


union_template = "input_sp{l}{r} = UNIONALL(input_sp{lm}{rm}, input_sp{i});"

if y==1:
  print "input = SCAN(trainingdata);"
else:
  print "input = SCAN(testdata);"

bwidth = 10
for i in range(nfeat):
  print parse_template.format(i=i, bwidth=bwidth)

print union_template.format(i=1, l=0, r=1, lm=0, rm='')

for i in range(2, nfeat-1):
  print union_template.format(i=i, l=0, r=i, lm=0, rm=i-1)

print union_template.format(i=nfeat-1, l='', r='', lm=0, rm=nfeat-2)
