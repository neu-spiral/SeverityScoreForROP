import xlrd as xr
import numpy as np

def loadComparisonData(name):
    """
    This function is to load and return each expert's comparison labels, and a dictionary that each ID links with the image order.

    Parameter:
    ----------
    name: expert's name

    Return:
    ---------
    IdOrder: dictionary
        the image ID in the comparison data and its corresponding order in the image feature
        Note: Image order starts from 1. But in python the order of each image feature starts from 0.
    comparisonData: 3 column matrix,
        1st column and 2nd column are the image ID in the comparison, the 3rd column is the comparison labels from the specific expert.

    """
    IdOrderFile = xr.open_workbook('../data/ropData/100Images.xlsx')
    IdorderSheet = IdOrderFile.sheet_by_name(u'ID&Order')
    Ids = IdorderSheet.col_values(0)
    del Ids[0]
    imageOrder = IdorderSheet.col_values(3)
    del imageOrder[0]
    IdOrder = dict(zip(Ids,imageOrder))
    temp = np.zeros((1,4))
    comparisonData = np.int_(temp)
    fCSV = open('../data/ropData/results_ICL_third_set_hundred_r2_compare.csv')
    for row in fCSV:
        if not name in row: continue
        content = row.split(',')
        number = content[0:4]
        number = [int(x) for x in number]
        comparisonData = np.vstack((comparisonData,np.reshape(number,[1,-1])))
    comparisonData=np.delete(comparisonData,0,0)
    comparisonData=np.delete(comparisonData,0,1)
    return IdOrder, comparisonData