#word cloud library courtesy of Andreas Mueller and MIT

import os
import pandas as pd
from os import path
from wordcloud import WordCloud, STOPWORDS

def flatten(l):
    stack = [iter(l)]
    result = []
    while stack:
        for item in stack[-1]:
            if isinstance(item, list):
                stack.append(iter(item))
                break
            result.append(item)
        else:
            stack.pop()
    return result

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
#d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

df = pd.read_csv('test.csv', usecols=['text'])
text_list = df.values.tolist()

# Read the whole text.
#text = open(path.join(d, 'constitution.txt')).read()

text = flatten(text_list)
text = " ".join(text)
more_stopwords = {"t", "co", "http", "https", "oh", "ÛÒ", "Û", "Û_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "shit", "fuck", "ass", "rt", "amp", "via", "get", "let"}
for item in more_stopwords:
    STOPWORDS.add(item)
# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# lower max_font_size
wordcloud = WordCloud(max_font_size=40, min_font_size=10, colormap= cm.cool , min_word_length=3).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()
