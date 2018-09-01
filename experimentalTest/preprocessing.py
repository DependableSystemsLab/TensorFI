# Import required libraries
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import sys, os, random 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


###
# Remove outliers in class label
def removeOutliers(data, yLabel):
	'''
	yFrequencyDic = {}
	for x in data.T:
		row_y = data.T[x][yLabel]
		if row_y not in yFrequencyDic:
			yFrequencyDic[row_y] = 0
		yFrequencyDic[row_y] += 1
	for x in data.T:
		row_y = data.T[x][yLabel]
		if yFrequencyDic[row_y] == 1:
			data = data.drop(x)
	return data
	'''		
	classCounts = data[yLabel].value_counts()
	for value, count in enumerate(classCounts):
		if count <= 1:
			data = data[data[yLabel] != classCounts.index.tolist()[value]]
	return data

###



###
# Convert categorical colum to int
class MultiColumnLabelEncoder:
	def __init__(self,columns = None):
        	self.columns = columns # array of column names to encode

	def fit(self,X,y=None):
        	return self # not relevant here

	def transform(self,X):
        	output = X.copy()
        	if self.columns is not None:
	        	for col in self.columns:
        	        	output[col] = LabelEncoder().fit_transform(output[col])
        	else:
            		for colname,col in output.iteritems():
                		output[colname] = LabelEncoder().fit_transform(col)
       		return output

	def fit_transform(self,X,y=None):
        	return self.fit(X,y).transform(X)

def convertObjectColum(data):
	cat_columns = data.select_dtypes(['object']).columns
	#data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
	data = MultiColumnLabelEncoder(columns = cat_columns).fit_transform(data)
	return data

###



###
# Fillup missing values

def fillUpMissingValues(data):
	data = data.dropna(how='any') # Drop rows where all cells in that row is NA
	data.fillna(0) # Fill in missing data with zeros
	return data

###


###
# Get the mapping between class label before and after preprocessing
def getMappingOfClass(originalData, cleanedData, yLabel):
	mapping = {}
	oD = originalData[yLabel].unique()
	cD = cleanedData[yLabel].unique()
	for x in range(0, len(oD)):
		# Given transformed label, report the original one
		mapping[str(cD[x])] = str(oD[x])
	return mapping

# Get the mapping of features
def getMappingOfFeatures(originalData, cleanedData, yLabel):
	mapping = {}
	# Only convert object type of columns
	catData = originalData.select_dtypes(['object'])
	for fname, values in catData.iteritems():
		# Skip class
		if fname == yLabel:
			continue
		# Features here
		oD = originalData[fname].unique()
		cD = cleanedData[fname].unique()
		mapping[fname] = {}
		for x in range(0, len(oD)):
			# Given original label, report the transformed one. This is different for y
			mapping[fname][str(oD[x])] = str(cD[x])
	return mapping

###





#########################
def cleanDataForClassification(data, yLabel):
	data = fillUpMissingValues(data)
	data = convertObjectColum(data)
	return data

def cleanDataForRegression(data, yLabel):
	data = fillUpMissingValues(data)
	data = convertObjectColum(data)
	return data
#########################
