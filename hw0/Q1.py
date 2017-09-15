import sys
count = 0
worddict = {}
wordlist = open(sys.argv[1]).read().split()
f=open('Q1.txt','w')
for w in wordlist:
    if w not in worddict:
        if count != 0:
            f.write("\n")
        
        f.write( "%s %d %d" % (w,count,wordlist.count(w)))
        count+=1