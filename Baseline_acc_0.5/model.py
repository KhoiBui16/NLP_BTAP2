import pickle
import numpy as np
import random

class FirstLM:
	def __init__(self):
		self.model = {"<END>": 0}
		
	def fit(self, data):
		ntoks = 0
		for s in data:
			for tok in s.split():
				if self.model.get(tok) == None:
					self.model[tok] = 1
				else:
					self.model[tok] += 1
				ntoks += 1
			self.model["<END>"] += 1
			
		for key in self.model:
			self.model[key] /= ntoks
					
	def generate(self):
		MAXLEN = 50
		TOKENS = []
		PROBS = []
		NTOKEN = len(self.model)
		ret = []
		for key in self.model:
			TOKENS.append(key)
			PROBS.append(self.model[key])
		
		c = 0
		for i in range(NTOKEN):
			c += PROBS[i]
			PROBS[i] = c
		for i in range(MAXLEN):
			p = random.uniform(0,1)
			for i in range(NTOKEN):
				if PROBS[i] >= p:
					break
			nextToken = ""
			if i < NTOKEN:
				nextToken = TOKENS[i]
			
			if nextToken == "<END>":
				if len(ret) > 1:
					break
				else:
					continue
			ret.append(nextToken)
		
		return " ".join(ret)
		
		
	def save(self):
		pickle.dump(self, open("FirstLM.mdl", "wb"))
	
class SecondLM:
	def __init__(self):
		self.model = {"<END>": 0}
		
	def fit(self, data):
		ntoks = 0
		for s in data:
			for tok in s.split():
				if self.model.get(tok) == None:
					self.model[tok] = 1
				else:
					self.model[tok] += 1
				ntoks += 1
			self.model["<END>"] += 1
			
		for key in self.model:
			self.model[key] /= ntoks
					
	def generate(self):
		MAXLEN = 50
		TOKENS = []
		PROBS = []
		NTOKEN = len(self.model)
		ret = []
		for key in self.model:
			TOKENS.append(key)
			PROBS.append(self.model[key])
		
		c = 0
		for i in range(NTOKEN):
			c += PROBS[i]
			PROBS[i] = c
		for i in range(MAXLEN):
			p = random.uniform(0,1)
			for i in range(NTOKEN):
				if PROBS[i] >= p:
					break
			nextToken = ""
			if i < NTOKEN:
				nextToken = TOKENS[i]
			
			if nextToken == "<END>":
				if len(ret) > 1:
					break
				else:
					continue
					
			ret.append(nextToken)
		
		return " ".join(ret)
		
		
	def save(self):
		pickle.dump(self, open("SecondLM.mdl", "wb"))
		