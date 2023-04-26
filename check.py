'''
Goal: checking results of post-processed model predictions

Metrices:
1. confusion matrix of ECAPs (NEI not included)
2. accuracy of all predicted pairs (NEI included)
3. claim-level label accuracy (voting)
'''

enum = {"Neither": 0, "True": 1, "False": 2}
