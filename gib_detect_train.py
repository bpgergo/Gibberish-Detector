#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import pickle
import collections

#Hungarian characters 
accepted_chars = u'abcdefghijklmnopqrstuvwxyzéáűőúöüóí '
pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])


def coroutine(func):
    """ A decorator function that takes care 
    of starting a coroutine automatically on call """
    def start(*args,**kwargs):
        coro = func(*args,**kwargs)
        coro.next()
        return coro
    return start


@coroutine
def filter_chars(accepted_chars,target):
    """ A coroutine to filter out unaccepted chars. 
    Accepts one char at a time  """
    while True:
        c = (yield)
        if c.lower() in accepted_chars:
            target.send(c.lower())

@coroutine
def ngrams(n, target):
    """ A coroutine to generate ngrams. 
    Accepts one char at a time """
    chars = collections.deque()
    while True:
        chars.append((yield))
        if len(chars) == n: 
            target.send(chars)
            chars.popleft()
        
@coroutine
def counter(matrix):
    """ A counter sink """
    while True:
        a, b = (yield)
        matrix[pos[a]][pos[b]] += 1

@coroutine
def rev_counter(matrix, res):
    """ A reverse counter sink """
    while True:
        a, b = (yield)
        res[0] += matrix[pos[a]][pos[b]]
        res[1] += 1


def train():
    """ Write a simple model as a pickle file """
    k = len(accepted_chars)
    enc  = "UTF-8"
    # Assume we have seen 10 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    counts = [[10 for i in xrange(k)] for i in xrange(k)]
    
    bigrams = filter_chars(accepted_chars, ngrams(2, counter(counts)))
    for c in open('big.txt').read().decode(enc): bigrams.send(c)
    
    # Normalize the counts so that they become log probabilities.  
    # We use log probabilities rather than straight probabilities to avoid
    # numeric underflow issues with long texts.
    # This contains a justification:
    # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
    for row in counts:
        s = float(sum(row))
        for j in xrange(len(row)):
            row[j] = math.log(row[j] / s)

    # Find the probability of generating a few arbitrarily choosen good and
    # bad phrases.
    good_probs = [avg_transition_prob(line, counts) \
        for line in open('good.txt').read().decode(enc).split('\n') if line]
    bad_probs = [avg_transition_prob(line, counts) \
        for line in open('bad.txt').read().decode(enc).split('\n') if line]
    # Assert that we actually are capable of detecting the junk.
    assert min(good_probs) > max(bad_probs)

    # And pick a threshold halfway between the worst good and best bad inputs.
    thresh = (min(good_probs) + max(bad_probs)) / 2
    pickle.dump({'mat': counts, 'thresh': thresh}, open('gib_model.pki', 'wb'))

def avg_transition_prob(line, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    res = [1.0, 0]
    bigrams = filter_chars(accepted_chars, ngrams(2, rev_counter(log_prob_mat, res)))    
    for c in line: bigrams.send(c)
    # The exponentiation translates from log probs to probs.
    return math.exp(res[0] / (res[1] or 1))

if __name__ == '__main__':
    train()
