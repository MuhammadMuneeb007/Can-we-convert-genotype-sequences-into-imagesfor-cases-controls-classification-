directory = sys.argv[1]

# Read the QC GWAS.
pvalues  = pd.read_csv(directory+os.sep+"files/Data.QC.gz",compression='gzip',sep="\t")
print(pvalues.head())


# The selection of p-value is explained in the manuscript. We started from the lower p-value threshold and then moved to significant p-value thresholds.
pvalue = float(sys.argv[2])
pvalues['P']=pd.to_numeric(pvalues['P'],errors='coerce')
subpvalues = pvalues[pvalues['P']<=float(sys.argv[2])]

# Make a directory in which sub-dataset will be stored.
if not os.path.isdir(directory+os.sep+"pv_"+str(float(sys.argv[2]))):
   os.mkdir(directory+os.sep+"pv_"+str(float(sys.argv[2])))

subpvalues.to_csv(directory+os.sep+"pv_"+str(float(sys.argv[2]))+os.sep+str(float(sys.argv[2]))+'.txt', columns=['SNP'],index=False,header=False)

# Extract selected SNPs from the test set.
os.system("./plink --bfile ./"+directory+os.sep+"test/test   --extract ./"+directory+os.sep+"pv_"+str(float(sys.argv[2]))+os.sep+str(float(sys.argv[2]))+".txt --recodeA --out "+directory+os.sep+"pv_"+str(float(sys.argv[2]))+os.sep+"ptest")


# Extract selected SNPs from the training set.
os.system("./plink --bfile ./"+directory+os.sep+"train/train   --extract ./"+directory+os.sep+"pv_"+str(float(sys.argv[2]))+os.sep+str(float(sys.argv[2]))+".txt --recodeA --out "+directory+os.sep+"pv_"+str(float(sys.argv[2]))+os.sep+"ptrain")
