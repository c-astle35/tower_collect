import re 

#create a SparkContext object from a donloaded .txt file of Shakespeare's Hamlet
doc = sc.textFile("///hdfs/user/lev/hamlet.txt")

#create an RDD that contains the words in a flat list
flat = doc.filter(lambda line: len(line)>0).flatMap(lambda line: re.split("\W+", line))

#create a key-value pair for each word and give each a value of 1
wordPairs = flat.filter(lambda word: len(word) > 0).map(lambda word: (word.lower(), 1))

#group together the same keys (words) which produce keys with lists of 1's summed together
wordcount = wordPairs.reduceByKey(lambda tuple1, tuple2: tuple1+tuple2)

#switch around the keys and their values, sort the tuples by keys with the greatest values being first
topWords = wordcount.map(lambda (w,c): (c,w)).sortByKey(ascending=False)

#show the top ten words used in Hamlet
topWords.take(10)
