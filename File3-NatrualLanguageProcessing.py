import nltk

#import brown corpus
from nltk.corpus import brown

#Make sure we get the right tagset
nltk.data.path.insert(0,'/group/ltg/projects/fnlp/nltk_data')

# module for training a Hidden Markov Model and tagging sequences
from nltk.tag.hmm import HiddenMarkovModelTagger

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist, LidstoneProbDist

### added:
from nltk.tag import map_tag
assert map_tag('brown','universal','NR-TL')=='NOUN','''
The installed Brown-to-Universal POS tag map is out of date.
Replace ~/nltk_data/taggers/universal_tagset/en-brown.map with
https://raw.githubusercontent.com/slavpetrov/universal-pos-tags/master/en-brown.map
'''
###

import operator
import random
from math import log
from numpy import argmin
from collections import defaultdict

class HMM:
  def __init__(self,train_data,test_data):
    self.train_data = train_data
    self.test_data = test_data
    self.states = []
    self.viterbi = []
    self.backpointer = []

  #compute emission model using ConditionalProbDist with the estimator: Lidstone probability distribution with +0.01 added to the sample count for each bin and an extra bin
  def emission_model(self,train_data):
    data = []
    for s in train_data:
        each = [(tag, word.lower()) for (word, tag) in s]
        data.extend(each)

    emission_FD = ConditionalFreqDist(data)
    estimator = lambda fd: LidstoneProbDist(fd, 0.01, fd.B() + 1)
    self.emission_PD = ConditionalProbDist(emission_FD, estimator)
    for tag,_ in data:
        if tag not in self.states:
            self.states.append(tag)
    print "states: ",self.states,"\n\n"
    #states:  [u'.', u'ADJ', u'ADP', u'ADV', u'CONJ', u'DET', u'NOUN', u'NUM', u'PRON', u'PRT', u'VERB', u'X']

    return self.emission_PD, self.states

  #test point 1a
  def test_emission(self):
    print "test emission"
    t1 = -self.emission_PD['NOUN'].logprob('fulton') #10.7862311423
    t2 = -self.emission_PD['X'].logprob('fulton') #-12.3247431105
    return t1,t2

  #compute transition model using ConditionalProbDist with the estimator: Lidstone probability distribution with +0.01 added to the sample count for each bin and an extra bin
  def transition_model(self,train_data):

    t = []
    for s in train_data:
      each = ["<s>"]
      each.extend([tag for (_, tag) in s])
      each.extend(["</s>"])
      t.extend(each)

    data = [(t[i], t[i+1]) for i in range(len(t) - 1)]

    transition_FD = ConditionalFreqDist(data)
    estimator = lambda fd: LidstoneProbDist(fd, 0.01, fd.B() + 1)
    self.transition_PD = ConditionalProbDist(transition_FD, estimator)

    return self.transition_PD

  #test point 1b
  def test_transition(self):
    print "test transition"
    transition_PD = self.transition_model(self.train_data)
    start = -transition_PD['<s>'].logprob('NOUN') #1.78408815305
    end = -transition_PD['NOUN'].logprob('</s>') #7.31426080296
    return start,end

  #train the HMM model
  def train(self):
    self.emission_model(self.train_data)
    self.transition_model(self.train_data)

  def set_models(self,emission_PD,transition_PD):
    self.emission_PD = emission_PD
    self.transition_PD = transition_PD

  #initialise data structures for tagging a new sentence
  #describe the data structures with comments
  #use the models stored in the variables: self.emission_PD and self.transition_PD
  #input: first word in the sentence to tag
  def initialise(self,observation):

    #We don't know the dimensions of the matrix at initialization (because we only recieve first word of sen)
    #So it we have to choose a dynamic structure so use defaultdicts as these are dynamic
    self.viterbi = defaultdict(lambda: defaultdict(lambda: 0))
    self.backpointer = defaultdict(lambda: defaultdict(lambda: 0))

    #initialise for transition from <s> , begining of sentence
    # use costs (-log-base-2 probabilities)
    for state in self.states:
        self.viterbi[state][0] = -self.transition_PD['<s>'].logprob(state) - self.emission_PD[state].logprob(observation)
        self.backpointer[state][0] = '<s>'
    #The viterbi is initialised by looping through the states and working out the probablitiy for the observation.
    #The backpointer is quite simply pointing to the start of the sentance.

  #tag a new sentence using the trained model and already initialised data structures
  #use the models stored in the variables: self.emission_PD and self.transition_PD
  #update the self.viterbi and self.backpointer datastructures
  #describe your implementation with comments
  #input: list of words
  def tag(self,observations):
    length = len(observations)

    for t in range(1,length):
      for state in self.states:
        potential_viterbi = []
        potential_back = []
        for i in range(len(self.states)): #Loop through all possiblities
            #t_value is the probalitiy of transitioning from one state to another
            #value is the probablity of transition and then emitting observation
            t_value = [self.viterbi[self.states[i]][t - 1] - self.transition_PD[self.states[i]].logprob(state)]
            value = [self.viterbi[self.states[i]][t - 1] - self.transition_PD[self.states[i]].logprob(state) - self.emission_PD[state].logprob(observations[t])]
            potential_viterbi.extend(value)
            potential_back.extend(t_value)
        self.viterbi[state][t] = min(potential_viterbi) #Get the most likely value for cell (Min because -log)
        self.backpointer[state][t] = self.states[argmin(potential_back)]

    potential_viterbi = []
    for i in range(len(self.states)):
        #value is probablity of transitioning from one state to </s>
        value = [self.viterbi[self.states[i]][length - 1] - self.transition_PD[self.states[i]].logprob("</s>")]
        potential_viterbi.extend(value)
    self.viterbi["</s>"][len(observations)] = min(potential_viterbi) #Same as above
    self.backpointer["</s>"][len(observations)] = self.states[argmin(potential_viterbi)]

    #reconstruct the tag sequence using the backpointer
    #return the tag sequence corresponding to the best path as a list (order should match that of the words in the sentence)
    tags = [None] * length #Initalise empty list
    current_back = self.backpointer["</s>"][length] #Backpointer for end of sentence
    tags[length - 1] = current_back #This will be the final tag
    for t in reversed(range(1, length)): #We start at end of sentence and work backwards hence loop from end
        tag = self.backpointer[current_back][t] #Find tag by just refering to backpointer of current
        tags[t - 1] = tag #Update list
        current_back = tag #Update current
    return tags

def answer_question4b():
    # Find a tagged sequence that is incorrect
    tagged_sequence = 'fixme'
    correct_sequence = 'fixme'
    # Why do you think the tagger tagged this example incorrectly?
    answer = 'fixme'
    return tagged_sequence, correct_sequence, answer

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def main():
  #devide corpus in train and test data
  tagged_sentences_Brown = brown.tagged_sents(categories= 'news')

  test_size = 1000
  train_size = len(tagged_sentences_Brown)-1000

  train_data_Brown = tagged_sentences_Brown[:train_size]
  test_data_Brown = tagged_sentences_Brown[-test_size:]

  tagged_sentences_Universal = brown.tagged_sents(categories= 'news', tagset='universal')
  train_data_Universal = tagged_sentences_Universal[:train_size]
  test_data_Universal = tagged_sentences_Universal[-test_size:]


  #create instance of HMM class and initialise the training and test sets
  obj = HMM(train_data_Universal,test_data_Universal)

  #train HMM
  obj.train()

  #part A: test emission model
  t1,t2 = obj.test_emission()
  print t1,t2
  if isclose(t1,10.7862311423) and isclose(t2,12.3247431105): ### updated again
    print "PASSED test emission\n"
  else:
    print "FAILED test emission\n"

  #part A: test transition model
  start,end = obj.test_transition()
  print start,end
  if isclose(start,1.78408815305) and isclose(end,7.31426080296):
    print "PASSED test transition\n"
  else:
    print "FAILED test transition\n"

  #part B: test accuracy on test set
  result = []
  correct = 0
  incorrect = 0
  accuracy = 0
  for sentence in test_data_Universal:
    s = [word.lower() for (word,tag) in sentence]
    obj.initialise(s[0])
    tags = obj.tag(s)
    for i in range(0,len(sentence)):
      if sentence[i][1] == tags[i]:
        correct+=1
      else:
        incorrect+=1
  accuracy = 1.0*correct/(correct+incorrect)
  print "accuracy: ",accuracy #accuracy:  0.857186331623
  if isclose(accuracy,0.857186331623): ### updated
    print "PASSED test viterbi\n"
  else:
    print "FAILED test viterbi\n"

  exit() # move me down as you fill in implementations


  # print answer for 4b
  tags, correct_tags, answer = answer_question4b()
  print("The incorrect tagged sequence is:")
  print(tags)
  print("The correct tagging of this sentence would be:")
  print(correct_tags)
  print("A possible reason why this error may have occured is:")
  print(answer[:280])

if __name__ == '__main__':

	main()
