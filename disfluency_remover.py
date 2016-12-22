import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-test', '--testfile', help='Name of file to be translated (one parse per line)', default='data/test.txt')
parser.add_argument('-train', '--trainfile', help='Name of file to output the translations to', default='data/train.txt')
parser.add_argument('-o', '--outfile', help='Name of output file', default='output.log')
parser.add_argument('-n', '--ngrams', help='Number of grams to use in this model', default=2)
args = parser.parse_args()

def main():
	train = args.trainfile
	test = args.testfile
	ngram = int(args.ngrams)
	if ngram == 1:
		print('Unigram Model:')
		unigram(train, test)
	elif ngram == 2:
		print('Bigram Model:')
		bigram(train, test)
	elif ngram == 3:
		print('Trigram Model:')
		trigram(train, test)
	else:
		print('Invalid ngram chosen. Please use either a unigram (1), a bigram (2), or a trigram (3) model')

def unigram(trainfile, testfile):
	outname = args.outfile
	outfile = open(outname, 'w+')
	counts = {} 	# dictionary containing a word as a key and a dictionary of tags and counts of the tags as a value
	tags = set()
	for line in open(trainfile):
		words  = line.strip('\n').split(' ')
		for w in words:
			pair = w.split('/')
			if len(pair) == 2:
				tags.add(pair[1])
				# keep track of word counts
				if pair[0] not in counts:
					counts[pair[0]] = {}
				if pair[1] not in counts[pair[0]]:
					counts[pair[0]][pair[1]] = 0
				counts[pair[0]][pair[1]] += 1
	# test model on test data and report accuracy
	correct = 0
	total = 0
	for line in open(testfile):
		words  = line.strip('\n').split(' ')
		output = ''
		for w in words:
			pair = w.split('/')
			if len(pair) == 2:
				total += 1
				guess = 'N'	# if not seen before, assume normal word
				# if seen, guess most common tag
				if pair[0] in counts:
					guess = get_max(counts[pair[0]])
				if guess == pair[1]:
					correct += 1
				# print just 'N' words to output log
				if guess == 'N':
					output += (pair[0] + ' ')
		outfile.write(output)
		outfile.write('\n')
	print('accuracy of unigram on test: ', correct/total)

def bigram(trainfile, testfile):
	outname = args.outfile
	outfile = open(outname, 'w+')
	# get tag-word and tag-tag counts
	all_tags = ['N', 'F', 'E', 'D', 'A', 'R', 'Y', 'Z']
	all_tps = get_tprime(1, all_tags, all_tags)
	tw, tt = ngram_counts(trainfile, 1)

	# compute tag-tag and word-tag probabilities
	ptt = {}
	ptw = {}
	for t in tw:
		ptw[t] = {}
		total = sum(tw[t].values())
		for w in tw[t]:
			if total > 0:
				ptw[t][w] = tw[t][w]/total
			else:
				ptw[t][w] = 0
	for t in tt:
		ptt[t] = {}
		total = sum(tt[t].values())
		for tp in tt[t]:
			if total > 0:
				ptt[t][tp] = tt[t][tp]/total
			else:
				ptt[t][tp] = 0

	# test model on test data
	total = 0
	correct = 0
	for line in open(testfile):
		output = ''
		line = line.strip('\n')
		w = line.split(' ')
		words = ['<s>']
		tags = ['Y']
		for pair in w:
			pair = pair.split('/')
			if len(pair) == 2:
				words.append(pair[0])
				tags.append(pair[1])
		words.append('</s>')
		tags.append('Z')
		viterbi, pointer = viter(ptw, ptt, words, all_tags, all_tps)
		# trace back through pointer to get best guess
		guess = ['Z']
		for i in range(len(words)-1,0,-1):
			x = pointer[i][guess[0]]
			guess.insert(0, x)
		for i in range(len(tags)):
			if words[i] not in ['<s>', '</s>']:
				if guess[i] == tags[i]:
					correct += 1
				# print just 'N' words to output log
				if guess[i] == 'N':
					output += words[i] + ' '
				total += 1
		outfile.write(output)
		outfile.write('\n')
	print('accuracy = ', correct/total)

def trigram(trainfile, testfile):
	outname = args.outfile
	outfile = open(outname, 'w+')
	# get tag-word and tag-tag counts
	all_tags = ['N', 'F', 'E', 'D', 'A', 'R', 'Y', 'Z']
	all_tps = new_get_tprime(2, all_tags, all_tags)
	tw, tt = new_ngram_counts(trainfile, 2, 0.0001)

	# compute tag-tag and word-tag probabilities
	ptt = {}
	ptw = {}
	for t in tw:
		ptw[t] = {}
		total = sum(tw[t].values())
		for w in tw[t]:
			if total > 0:
				ptw[t][w] = tw[t][w]/total
			else:
				ptw[t][w] = 0
	for t in tt:
		ptt[t] = {}
		total = sum(tt[t].values())
		for tp in tt[t]:
			if total > 0:
				ptt[t][tp] = tt[t][tp]/total
			else:
				ptt[t][tp] = 0

	# test model on training data
	total = 0
	correct = 0
	count_lines = sum(1 for line in open(testfile))
	curr = 0
	perc = 0
	for line in open(testfile):
		line = line.strip('\n')
		w = line.split(' ')
		words = ['<s>']
		tags = ['YY']
		output = ''
		for pair in w:
			pair = pair.split('/')
			if len(pair) == 2:
				words.append(pair[0])
				tags.append(tags[-1][-1]+pair[1])
		words.append('</s>')
		tags.append('ZZ')
		viterbi, pointer = viter(ptw, ptt, words, all_tps, all_tps)
		# trace back through pointer to get best guess
		guess = ['ZZ']
		for i in range(len(words)-1,0,-1):
			x = pointer[i][guess[0]]
			guess.insert(0, x)
		for i in range(len(tags)):
			if words[i] not in ['<s>', '</s>']:
				if guess[i][-1] == tags[i][-1]:
					correct += 1
				total += 1
				if guess[i][-1] == 'N':
					output += words[i] + ' '
		outfile.write(output)
		# running percentages to know how much time is left
		curr += 1
		if perc != int(curr*100/count_lines):
			perc = int(curr*100/count_lines)
			print( perc, ' % complete')
			print('{0:.2f}'.format(correct*100/total), ' % accurate')
	print('accuracy = ', correct/total)

	## 3.4 - show output on second line of testing data
	#line = open(testfile).readlines()[1].strip('\n')
	#w = line.split(' ')
	#words = ['<s>']
	#tags = ['YY']
	#for pair in w:
	#	pair = pair.split('/')
	#	if len(pair) == 2:
	#		words.append(pair[0])
	#		tags.append(tags[-1][-1]+pair[1])
	#words.append('</s>')
	#tags.append('ZZ')
	#viterbi, pointer = viter(ptw, ptt, words, all_tps, all_tps)
	## trace back through pointer to get best guess
	#guess = ['ZZ']
	#output = '</s>/ZZ'
	#s = ''
	#for i in range(len(words)-1,0,-1):
	#	x = pointer[i][guess[0]]
	#	guess.insert(0, x)
	#	output =  words[i-1]+ '/' + guess[0][-1] + ' ' + output
	#for i in range(len(words)):
	#	if guess[i][-1] == 'N':
	#		s += words[i] + ' '

	## compute the log probability of the second line
	#Pwt = 1
	#Pt = ptt[tags[0]]['YY']
	#for w in range(len(words)):
	#	if words[w] in ptw[tags[w]]:
	#		Pwt *= ptw[tags[w]][words[w]]
	#	else:
	#		Pwt *= 1
	#		Pwt *= ptw[tags[w]]['<unk>']
	#	if w > 0:
	#		Pt *= ptt[tags[w]][tags[w-1]]
	#Pt *= ptt['ZZ'][tags[-1]]
	#print('log probability = ', math.log(Pt*Pwt))
	#print(output)
	#print(s)

def viter(tw, tt, words, all_tags, all_tps):	# actual viterbi algorithem itself
	# viterbi[q] = 0 for q!= q0
	gram_len = len(all_tps[0])
	viterbi = {}
	for w in range(len(words)):
		viterbi[w] = {}
		#for t in all_tags:
		for t in all_tps:
			viterbi[w][t] = 0
	# viterbi[q0] = 1
	s = 'Y'*gram_len
	viterbi[0][s] = 1
	pointer = {}
	# for each state q' in topological order (q' = [q][t] here)
	# q ensures topological order, q and t combine to make the current state
	for q in range(1,len(words)):
		#for t in all_tags:
		for t in all_tps:
			# hacky - fix it with smoothing, assumes if all weights are 0 that 'N' is best
			if q not in pointer:
				pointer[q] = {}
			pointer[q][t] = 'N'*gram_len
			# for each incoming transition q->q' (q = [q-1][i] here)
			#for i in all_tags:
			for i in all_tps:
				# with weight p
				p = tt[t][i]*tw[t]['<unk>']
				# for some reason the smoothing decreases the accuracy from ~90% to ~77% so I kept the line below, which gets rid of the smoothing...
				p = 0
				if words[q] in tw[t]:
					p = tt[t][i]*tw[t][words[q]] 
				# if viterbi[q]*p > viterbi[q']
				if viterbi[q-1][i]*p > viterbi[q][t]:
					# viterbi[q'] = viterbi[q]*p
					viterbi[q][t] = viterbi[q-1][i]*p
					# pointer[q'] = q
					if q not in pointer:
						pointer[q] = {}
					pointer[q][t] = i
	return viterbi, pointer

def new_get_tprime(n, tags, all_tags):
	# base case
	if n == 1:
		return all_tags
	# go another level
	new_t = []
	for tp in all_tags:
		for t in tags:
			new_t.append(tp+t)
	# recurse
	return new_get_tprime(n-1, tags, new_t)

def get_tprime(n, tags, all_tags):
	# base case
	if n == 1:
		return all_tags
	# go another level
	new_t = []
	for tp in all_tags:
		for t in tags:
			new_t.append(tp+' '+t)
	# recurse
	return get_tprime(n-1, tags, new_t)

def new_ngram_counts(filename, gram_len, d):
	all_tags = ['Y', 'N', 'F', 'E', 'D', 'A', 'R', 'Z']
	all_tps = new_get_tprime(gram_len, all_tags, all_tags)
	model = {} 	# for p(w  |  t')
	count = {}	# for p(t' |  t )
	delta = d
	# smoothing p(t'|t) and p(w|t')
	for t in all_tps:
		model[t] = {}
		model[t]['<unk>'] = delta
		count[t] = {}
		for tp in all_tps:
			count[t][tp] = delta
	for line in open(filename):
		# set up words and tags arrays
		line = line.strip('\n')
		w = line.split(' ')
		words = ['<s>' for i in range(gram_len-1)]
		tags = ['Y'*gram_len for i in range(gram_len-1)]
		for pair in w:
			pair = pair.split('/')
			if len(pair) == 2:
				words.append(pair[0])
				if gram_len > 1:
					tags.append(tags[-1][-1] + pair[1])
				else:
					tags.append(pair[1])
		words.extend(['</s>' for i in range(gram_len-1)])
		tags.extend(['Z'*gram_len for i in range(gram_len-1)])
		for w in range(len(words)):
			# get counts for t' and t
			if w > 0:
				if tags[w] not in count:
					count[tags[w]] = {}
				if tags[w-1] not in count[tags[w]]:
					count[tags[w]][tags[w-1]] = 0
				count[tags[w]][tags[w-1]] += 1
			# get counts for t and w
			if tags[w] not in model:
				model[tags[w]] = {}
			if words[w] not in model[tags[w]]:
				model[tags[w]][words[w]] = 0
			model[tags[w]][words[w]] += 1
	return model, count
		
def ngram_counts(filename, gram_len):
	all_tags = ['Y', 'N', 'F', 'E', 'D', 'A', 'R', 'Z']
	all_tps = get_tprime(gram_len, all_tags, all_tags)
	model = {} 	# for p(w  |  t')
	count = {}	# for p(t' |  t )
	delta = 0.01
	# smoothing p(t'|t) and p(w|t')
	for t in all_tags:
		model[t] = {}
		model[t]['<unk>'] = delta
		count[t] = {}
		for tp in all_tps:
			count[t][tp] = delta
	for line in open(filename):
		line = line.strip('\n')
		w = line.split(' ')
		words = []
		tags = []
		for pair in w:
			pair = pair.split('/')
			if len(pair) == 2:
				words.append(pair[0])
				tags.append(pair[1])
		## <s> tags ##
		# set counts in model
		if "Y" not in model:
			model["Y"] = {}
		if "<s>" not in model["Y"]:
			model["Y"]["<s>"] = 0
		model["Y"]["<s>"] += gram_len
		# set counts in count
		for i in range(gram_len):
			y = gram_len-i-1
			if len(tags) <= y:
				s = ' '.join(['Y' for x in range(i+1)] + tags + ['Z' for y in range(gram_len-i-len(tags))])
				if 'Z' not in count:
					count['Z'] = {}
				if s not in count['Z']:
					count['Z'][s] = 0
				count['Z'][s] +=1
			else:	
				s = ' '.join(['Y' for x in range(i+1)] + tags[:y])
				if tags[y] not in count:
					count[tags[y]] = {}
				if s not in count[tags[y]]:
					count[tags[y]][s] = 0
				count[tags[y]][s] +=1
		## middle section ##
		for w in range(len(words)):
			y = w+gram_len
			# set counts in model
			if tags[w] not in model:
				model[tags[w]] = {}
			if words[w] not in model[tags[w]]:
				model[tags[w]][words[w]] = 0
			model[tags[w]][words[w]] += 1
			# set counts in count
			if y < len(words):
				t_prime = ' '.join(tags[w:y])
				if tags[y] not in count:
					count[tags[y]] = {}
				if t_prime not in count[tags[y]]:
					count[tags[y]][t_prime] = 0
				count[tags[y]][t_prime] += 1
		## </s> tags ##
		# set counts in model
		if "Z" not in model:
			model["Z"] = {}
		if "</s>" not in model["Z"]:
			model["Z"]["</s>"] = 0
		model["Z"]["</s>"] += gram_len
		# set counts in count
		for i in range(gram_len):
			y = -1*(gram_len-i)
			s = ' '.join(tags[y:] + ['Z' for x in range(i)])
			if 'Z' not in count:
				count['Z'] = {}
			if s not in count['Z']:
				count['Z'][s] = 0
			count['Z'][s] +=1
	return model, count

def get_max(d):
	max_v = 0
	max_k = ''
	for k in d:
		if d[k] > max_v:
			max_k = k
			max_v = d[k]
	return max_k

if __name__ == '__main__':
	main()
