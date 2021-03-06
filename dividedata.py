import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sys

def reformat():
  pass
  

def subsubsection(pheno,direc,name):
    case = pheno.loc[pheno["phenotype"]==1]  
    control = pheno.loc[pheno["phenotype"]==0]
    case.to_csv(direc+os.sep+name+"_case_id.txt", index=False, columns=['user_id'],header=False)
    control.to_csv(direc+os.sep+name+"_control_id.txt", index=False, columns=['user_id'],header=False)
    pheno.to_csv(direc+os.sep+name+"_id.txt", index=False, columns=['user_id'],header=False)
  
  
  
def saveandformatsamplefile(top,bottom,direc):
 
    phenotype = pd.DataFrame()
    phenotype["user_id"] = bottom["ID_1"].values
    phenotype["phenotype"] = bottom["pheno"].values
    phenotype.to_csv(direc+os.sep+"phenotype.csv",sep="\t",index=False)
    
  ###make covarite file
    cov  = pd.DataFrame()
    cov["FID"] = bottom["ID_2"].values
    cov["IID"] = bottom["ID_1"].values
    cov["Sex"] = 1
    cov["cov1"] = bottom["sharedConfounder1_bin1"].values 
    cov["cov2"] = bottom["independentConfounder2_cat_norm1"].values
    cov["cov3"] = bottom["independentConfounder2_cat_norm2"].values
    cov["cov4"] = bottom["independentConfounder3_cat_unif1"].values
    sampletop  =  top.copy()
    samplebottom = bottom.copy()

    samplebottom["pheno"] = samplebottom["pheno"].apply(pd.to_numeric)
    samplebottom.pheno[samplebottom['pheno']<0]=0
    samplebottom.pheno[samplebottom['pheno']>0]=1
    samplebottom["pheno"] = pd.to_numeric(samplebottom["pheno"],downcast='integer')
    sample = pd.concat([sampletop, samplebottom], axis=0)
    sample= sample.astype(str)
    
    
    if "test" in direc:
        subsubsection(phenotype,direc,"test")
        sample.to_csv(direc+os.sep+"test_snptest.sample",index=False,sep=" ")
    
    if "train" in direc:
        subsubsection(phenotype,direc,"train")
        sample.to_csv(direc+os.sep+"train_snptest.sample",index=False,sep=" ")
    
    
    
    
    
    
    sample.pheno[sample['pheno']=='1']='2'
    sample.pheno[sample['pheno']=='0']='1'

    data = sample[["ID_1","ID_2","missing","pheno"]]
    if "test" in direc:
        data.to_csv(direc+os.sep+"test.sample",index=False,sep=" ")
    
    if "train" in direc:
        data.to_csv(direc+os.sep+"train.sample",index=False,sep=" ")

    samplebottom.pheno[samplebottom['pheno']==1]='2'
    samplebottom.pheno[samplebottom['pheno']==0]='1'

    cov["cov5"] = bottom["sharedConfounder4_norm1"].values
    cov["cov6"] = bottom["independentConfounder4_norm1"].values
    cov.to_csv(direc+os.sep+"YRI.covariate",index=False,sep="\t")
    ###PRS phenotype
    phenotype = pd.DataFrame()
    phenotype["FID"] = bottom["ID_2"].values
    phenotype["IID"] = bottom["ID_1"].values
    phenotype["phenotype"] = bottom["pheno"].values
    phenotype.to_csv(direc+os.sep+"YRI.pheno",sep="\t",index=False)
    ###NewSample file
    sample = pd.concat([top, bottom], axis=0)
    sample = sample[['ID_1','ID_2',  'missing','pheno']]
    sample.to_csv(direc+os.sep+"YRIs.sample",index=False,sep=" ")
    return phenotype['FID'].values


    
def splitsample(sample,direc):
    sampletop  = sample.head(1)
    samplebottom = sample.tail(len(sample)-1)
    samplebottom['ID_1'] = samplebottom['ID_1'].astype(str)+str("_") + samplebottom['ID_1'].astype(str)
    samplebottom['ID_2'] = samplebottom['ID_2'].astype(str)+str("_") + samplebottom['ID_2'].astype(str)

    samplebottom["pheno"] = samplebottom["pheno"].apply(pd.to_numeric)
    samplebottom["pheno"].values[samplebottom["pheno"] < 0] = 0
    samplebottom["pheno"].values[samplebottom["pheno"] > 0] = 1
    samplebottom["pheno"] = pd.to_numeric(samplebottom["pheno"],downcast='integer')
    x_train, x_test, y_train, y_test = train_test_split(samplebottom, samplebottom["pheno"].values)
    sampletop.iloc[0,9]="B"
    trainsample = saveandformatsamplefile(sampletop, x_train,direc+os.sep+"train")
    testsample = saveandformatsamplefile(sampletop, x_test,direc+os.sep+"test")
    return trainsample,testsample
 
def commit(direc,name):
  sample  = pd.read_csv(direc+os.sep+name+".sample",sep=" ")
  samplebottom = sample.tail(len(sample)-1)
  fam = pd.read_csv(direc+os.sep+name+".fam",sep="\s+",header=None)
  fam[5] = samplebottom['pheno'].values
  fam.to_csv(direc+os.sep+name+".fam",header=False,index=False, sep=" ")
    


# Directory name in which files will be stored.
# Create four directories to contain train, test, and intermediate files.

direc = sys.argv[1]
if not os.path.isdir(direc):
  os.mkdir(direc)
if not os.path.isdir(direc+os.sep+"test"):
  os.mkdir(direc+os.sep+"test")
if not os.path.isdir(direc+os.sep+"train"):
  os.mkdir(direc+os.sep+"train")
if not os.path.isdir(direc+os.sep+"files"):
  os.mkdir(direc+os.sep+"files")

testdirec = direc+os.sep+"test"
traindirec = direc+os.sep+"train"
filesdirec = direc+os.sep+"files"

# Read the sample files, and ensure path is correct.


originalsamples = pd.read_csv("/l/proj/kuin0009/MuhammadMuneeb/MM_images2/generatedata/CEU_merge/Ysim_snptest.sample",sep=" ")

# This function splits the samples into training and test sets.
train,test = splitsample(originalsamples,direc)

# Extract test samples using bcftools
os.system("bcftools view -S ./"+testdirec+os.sep+"test_id.txt /l/proj/kuin0009/MuhammadMuneeb/MM_images2/generatedata/CEU_merge/X.vcf  > ./"+testdirec+os.sep+"test.vcf")
os.system(" ./plink --vcf ./"+testdirec+os.sep+"test.vcf --make-bed --out ./"+testdirec+os.sep+"test")
os.system("./plink --bfile  ./"+testdirec+os.sep+"test  --recode --tab --out ./"+testdirec+os.sep+"test")


# Extract training samples using bcftools
os.system("bcftools view -S ./"+traindirec+os.sep+"train_id.txt /l/proj/kuin0009/MuhammadMuneeb/MM_images2/generatedata/CEU_merge/X.vcf  > ./"+traindirec+os.sep+"train.vcf")
os.system("./plink --vcf ./"+traindirec+os.sep+"train.vcf --make-bed  --out ./"+traindirec+os.sep+"train")


# Modify the fam file.
commit(testdirec,"test")
commit(traindirec,"train")

os.system("./plink --bfile  ./"+traindirec+os.sep+"train  --recode --tab --out ./"+traindirec+os.sep+"train")

# Calculate GWAS using plink
os.system("./plink --bfile ./"+traindirec+os.sep+"train --allow-no-sex --fisher --out ./"+traindirec+os.sep+"train")
