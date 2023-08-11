# We use spaCy to handle the basics such as tokenization.
import spacy
nlp = spacy.load("en_core_web_sm")

import numpy as np
import re
from collections import defaultdict


class NgramLanguageModel(object):

    def __init__(self, n, sos='<s>', eos='</s>'):
        self.n, self.sos, self.eos = n, sos, eos

        # We need at least bigrams for this model to work
        if self.n < 2:
            raise Exception('Size of n-grams must be at least 2!')
        
        # keeps track of how many times ngram has appeared in the text before
        self.ngram_counter = defaultdict(int)

        # Dictionary that keeps list of candidate words given context
        # When generating a text, we only pick from those candidate words
        self.context = {}

        
    def preprocess_sentence(self, s, lowercase=True):
        # Do some cleaning
        s = s.encode("ascii", "ignore").decode()
        s = re.sub(r'\s+', ' ', s)    

        # Do case folding to lowercase if specified
        if lowercase == True:
            s = s.lower()

        # Tokenize sentence and return list of tokens
        return [ t.text.strip() for t in nlp(s) if t.text.strip() != '']        

    
    def get_ngrams(self, tokens: list) -> list:
        """
        Generates all n-grams of size n given a list of tokens
        :param tokens: tokenized sentence
        :return: list of ngrams
        ngrams of tuple form: ((previous wordS!), target word)
        """
        n = self.n
        
        tokens = (n-1)*[self.sos] + tokens + (n-1)*[self.eos]
        l = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))]
        return l
        
        
    def update(self, s: str) -> None:
        """
        Updates Language Model
        :param sentence: input text
        """
        tokens = self.preprocess_sentence(s)
        
        if len(tokens) < self.n:
            return
        
        ngrams = self.get_ngrams(tokens)
        for ngram in ngrams:
            self.ngram_counter[ngram] += 1.0
            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word)
            else:
                self.context[prev_words] = [target_word]

                
    def calc_prob(self, context, token):
        """
        Calculates probability of a token given a context
        :return: conditional probability
        """
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.context[context]))
            result = count_of_token / count_of_context
        except KeyError:
            result = 0.0
        return result

    
    def random_token(self, context):
        """
        Given a context we "semi-randomly" select the next word to append in a sequence
        :param context:
        :return:
        """
        # Get all candidate words for the given context
        tokens_of_interest = self.context[context]
        # Get the probavilities for each ngram (context+word)
        token_probs = np.array([self.calc_prob(context, token) for token in tokens_of_interest])
        # Return a random candidate word based on the probability distribution
        # (candidate words that occur more frequently after the context have a higher prob)
        return np.random.choice(tokens_of_interest, 1, p=(token_probs / sum(token_probs)))[0]


            
    def generate_text(self, token_count: int, start_context=[]):
        """
        Iteratively generates a sentence by predicted the next word step by step
        :param token_count: number of words to be produced
        :param start_context: list of start/seed words
        :return: generated text
        """
        n = self.n
        
        # The following block merely prepares the first context; note that the context is always of size
        # (self.n - 1) so depending on the start_context (representing the start/seed words), we need to
        # pad or cut off the start_context.
        if len(start_context) == (n-1):
            context_queue = start_context.copy()
        elif len(start_context) < (n-1):
            context_queue = ((n - (len(start_context)+1)) * [self.sos]) + start_context.copy()
        elif len(start_context) > (n-1):
            context_queue = start_context[-(n-1):].copy()
        result = start_context.copy()                    
            
        # The main loop for generating words
        for _ in range(token_count):
            # Generate the next token given the current context
            obj = self.random_token(tuple(context_queue))
            # Add generated word to the result list
            result.append(obj)
            # Remove the first token from the context
            context_queue.pop(0)
            if obj == self.eos:
                # If we generate the EOS token, we can return the sentence (without the EOS token)
                return ' '.join(result[:-1])
            else:
                # Otherwise create the new context and keep generate the next word
                context_queue.append(obj)
        # Fallback if we predict more than token_count tokens
        return ' '.join(result)             