import sys

file = open(sys.argv[2],'r')
lines = file.readlines()
elements = []
for ll in lines:
    elements.append(ll.split())

newLine = []
for i in range(len(elements)):
    newLine.append(float(elements[i][int(sys.argv[1])]))
newLine.sort()

ans=""
for i in range(len(newLine)):
    if i==0:
        ans = str(newLine[i])
    else:
        ans = ans + ','+ str(newLine[i])

#print ans
output = open('ans1.txt','w')
output.write(ans)

file.close()
output.close()
    


