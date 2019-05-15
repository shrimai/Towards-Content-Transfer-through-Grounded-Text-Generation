###
# This code is based on the SumBasic implementation of https://github.com/EthanMacdonald/SumBasic
###

import nltk, glob, codecs
import argparse, csv, ast, os

lemmatize = True
rm_stopwords = True
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

parser = argparse.ArgumentParser(description='sumbasicUpdate.py')
parser.add_argument('-output', required=True,
                    help="Path to the output file")
parser.add_argument('-input', required=True,
                    help="Path to the input csv file")
parser.add_argument('-num_sentences', default=1,
                    help="Path to the input csv file")
parser.add_argument('-context_update', action='store_true',
                    help="whether you want to force decode or not")
opt = parser.parse_args()


def clean_sentence(tokens):
	tokens = [t.lower() for t in tokens]
	if lemmatize: tokens = [lemmatizer.lemmatize(t) for t in tokens]
	if rm_stopwords: tokens = [t for t in tokens if t not in stopwords]
	return tokens

def get_probabilities(cluster, lemmatize, rm_stopwords):
	# Store word probabilities for this cluster
	word_ps = {}
	# Keep track of the number of tokens to calculate probabilities later
	token_count = 0.0
	# Gather counts for all words in all documents
	for path in cluster:
		with codecs.open(path, "r", "utf-8", errors="ignore") as f:
			tokens = clean_sentence(nltk.word_tokenize(f.read()))
			token_count += len(tokens)
			for token in tokens:
				if token not in word_ps:
					word_ps[token] = 1.0
				else:
					word_ps[token] += 1.0
	# Divide word counts by the number of tokens across all files
	for word_p in word_ps:
		word_ps[word_p] = word_ps[word_p]/float(token_count)
	return word_ps

def get_sentences(cluster):
	sentences = []
	for path in cluster:
		with codecs.open(path, "r", "utf-8", errors="ignore") as f:
			sentences += nltk.sent_tokenize(f.read())
	return sentences

def clean_sentence(tokens):
	tokens = [t.lower() for t in tokens]
	if lemmatize: tokens = [lemmatizer.lemmatize(t) for t in tokens]
	if rm_stopwords: tokens = [t for t in tokens if t not in stopwords]
	return tokens

def score_sentence(sentence, word_ps):
	score = 0.0
	num_tokens = 0.0
	sentence = nltk.word_tokenize(sentence)
	tokens = clean_sentence(sentence)
	for token in tokens:
		if token in word_ps:
			score += word_ps[token]
			num_tokens += 1.0
	if num_tokens == 0.0:
		return score
	return float(score)/float(num_tokens)

def get_max_sentence(sentences, word_ps, simplified):
	max_sentence = None
	max_score = None
	for sentence in sentences:
		score = score_sentence(sentence, word_ps)
		if max_score == None or score > max_score:
			max_sentence = sentence
			max_score = score
	if not simplified: 
		word_ps = update_ps(max_sentence, word_ps)
	return max_sentence, word_ps

def update_ps(max_sentence, word_ps):
	sentence = nltk.word_tokenize(max_sentence)
	sentence = clean_sentence(sentence)
	for word in sentence:
		if word in word_ps:
			word_ps[word] = word_ps[word]**2
	return word_ps

def force_decode(sentences, word_ps):
	try:
		index = sentences.index("WIKIPEDIA CONTEXT.")
	except:
		t = [sent for sent in sentences if "WIKIPEDIA CONTEXT." in sent]
		ti = sentences.index(t[0])
		s = t[0].split("WIKIPEDIA CONTEXT.")[0]
		sentences = sentences[:ti] + [s] + ["WIKIPEDIA CONTEXT."] + sentences[ti+1:]
		index = ti + 1
	for sent in sentences[index+1:]:
		word_ps = update_ps(sent, word_ps)
	return sentences[:index]

def orig(cluster):
	cluster = glob.glob(cluster)
	word_ps = get_probabilities(cluster, lemmatize, rm_stopwords)
	sentences = get_sentences(cluster)
	#print(sentences)
	if opt.context_update:
		sentences = force_decode(sentences, word_ps)
	summary = []
	for i in range(opt.num_sentences):
		max_sentence, word_ps = get_max_sentence(sentences, word_ps, False)
		summary.append(max_sentence)
	return " ".join(summary)

def main():
	outF = codecs.open(opt.output, "w", "utf-8")
	count = 0
	with codecs.open(opt.input, "r", "utf-8") as inp:
		spam = csv.reader(inp, delimiter=',')
		for row in spam:
			count += 1
			### Skip the header line
			if count == 1:
				continue
			news_article = row[10].split("\n")
			wiki_context = ast.literal_eval(row[4])
			try:
				news_article = [item for part in news_article for item in sent_tokenize(part)]
			except:
				pass

			news_article = '\n\n'.join(news_article)
			wiki_context = '\n\n'.join(wiki_context)
			#data = unicode(data, errors="ignore")
			with codecs.open("tmpD.txt", "w", "utf-8") as outmp:
				outmp.write(news_article + "\n")
				if opt.context_update:
					outmp.write("\nWIKIPEDIA CONTEXT.\n\n")
					outmp.write(wiki_context + "\n")

			summary = orig("tmpD.txt")
			#print(summary)
			summary = ' '.join(summary.strip().split("\n"))
			outF.write(summary + "\n")
	outF.close()
	os.remove("tmpD.txt")
	print("tmpD.txt Removed!")

if __name__ == '__main__':
	main()
