import numpy as np

data = np.loadtxt(open("../ML-Proj_dataset/kddcup.data_10_percent_corrected", "rb"), dtype=str, delimiter=",")

n,m =len(data), len(data[0])

answer=np.zeros(n)
newdata= np.zeros((n, m+7))
for i in range (n):
	
	newdata[i][0]=data[i][0]

	if data[i][1]=="icmp" :
		newdata[i][1]='1'
	if data[i][1]=="tcp" :
		newdata[i][2]='1'
	if data[i][1]=="udp" :
		newdata[i][3]='1'
	if data[i][2]=="eco_i" :
		newdata[i][4]=1
	if data[i][2]=="urp_i" :
		newdata[i][5]='1'
	if data[i][2]=="ecr_i" :
		newdata[i][6]='1'
	if data[i][2]=="private" :
		newdata[i][7]='1'
	if data[i][2]=="other" :
		newdata[i][8]='1'
	if data[i][3]=="SF" :
		newdata[i][9]='1'
	if data[i][3]=="S0" :
		newdata[i][10]='1'

	for j in range(4,m-1):
		newdata[i][j+7]=data[i][j]
	
	if data[i][m-1]!='normal':
		answer[i]=1

newdata=newdata.astype(np.float)

print(newdata[0])
