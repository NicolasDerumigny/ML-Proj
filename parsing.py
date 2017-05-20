import numpy as np

def load():
	data = np.loadtxt(open("../ML-Proj_dataset/kddcup.data_10_percent_corrected", "rb"), dtype=str, delimiter=",")

	n,m =len(data), len(data[0])

	answer=np.zeros(n)
	training=[]
	newdata= np.zeros((n, m+7))
	for i in range (n):
		
		newdata[i][0]=float(data[i][0][2:-1])
		if data[i][1]=="b'icmp'" :
			newdata[i][1]=1
		if data[i][1]=="b'tcp'" :
			newdata[i][2]=1
		if data[i][1]=="udp'" :
			newdata[i][3]=1
		if data[i][2]=="b'eco_i'" :
			newdata[i][4]=1
		if data[i][2]=="b'urp_i'" :
			newdata[i][5]=1
		if data[i][2]=="b'ecr_i'" :
			newdata[i][6]=1
		if data[i][2]=="b'private'" :
			newdata[i][7]=1
		if data[i][2]=="b'other'" :
			newdata[i][8]=1
		if data[i][3]=="b'SF'" :
			newdata[i][9]=1
		if data[i][3]=="b'S0'" :
			newdata[i][10]=1

		for j in range(4,m-1):
			newdata[i][j+7]=float(data[i][j][2:-1])

		
		if data[i][m-1]!="b'normal.'":
			answer[i]=1
		else:
			training+=[newdata[i]]

	return training, newdata, answer