import re

from bs4 import BeautifulSoup
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)


class Preprocessor(object):

    def __init__(self, remove_tags=True, clean_text=False, lemma=False, stopwords=False):
        self.remove_tags_func = self.remove_tags if remove_tags else lambda x: x
        self.clean_text_func = self.clean_text if clean_text else lambda x: x
        self.lemma_func = self.lemma if lemma else lambda x: x
        self.stopwords_func = self.remove_stopwords if stopwords else lambda x: x


    def remove_tags(self, text):
        normalizer = re.compile("<\\s*img\\b|<\\s*image|<\\s*span\\b|<\\W*?[phPH].*?>", re.UNICODE)
        text = normalizer.sub(lambda x: ' ' + x.group(0), text)
        soup = BeautifulSoup(text, 'html.parser')

        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text


    def clean_text(self, text):
        text = re.sub(r'[?|!|\'|"|#]', " ", text)
        text = re.sub(r'[,|.|;|:|(|)|{|}|\|/|<|>|-]', " ", text)
        text = re.sub('[^а-яёА-ЯЁ a-zA-Z]+', " ", text)

        return text


    def lemma(self, text):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        tokens_list = []
        for tok in doc.tokens:
            tok.lemmatize(morph_vocab)
            tokens_list.append(tok.lemma)
        
        return tokens_list


    def remove_stopwords(self, text):

        file = open('corus_rubrication/src/stopwords_russian.txt', 'r')
        content = file.read()
        content_list = content.split('\n')
        file.close()
    
        stop_words = set(content_list)
        no_stop_words = [word for word in text if word not in stop_words]
        no_stop_words = [word for word in no_stop_words if len(word)>=3]
        no_stop_text = ' '.join(no_stop_words)
        return no_stop_text




    def __call__(self, text):
        text = self.remove_tags_func(text)
        text = self.clean_text_func(text)
        text = self.lemma_func(text)
        text = self.stopwords_func(text)
        text = re.sub(r'^\s*|\s\s*', ' ', text).strip()
        return text