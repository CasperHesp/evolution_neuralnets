import numpy as np
import matplotlib.pyplot as plt
timesteps = 25

size = 10
M = .5
dmax = 100
Smax = 25
dt = 1
Estart = 50
inert = 0.4*M*size**2
wtra = 0.02
wrot = 0.02

nagent = 2
nfood = 10
npred = 5
nscent = 3

b = 0.1
ni = 6
nh = 6
nc = 6
no = 3

def sigmoid(X):
	if len(np.atleast_1d(X))>1:
		Y = np.zeros(len(X))
		for i in range(len(X)):
			if not (X[i]<0):	
				Y[i] = X[i]*(1+X[i])**-1 #(1+exp(-1*X))**-1
	else:
		Y = 0
		if not (X<0):
			Y = X*(1+X)**-1
	return Y

I = np.zeros(ni)
H = np.zeros(nh)
C = np.zeros(nc)
O = np.zeros(no)

#Hiw = np.zeros((ni/2,nh))
#Hcw = np.zeros((nc/2,nh))
#Chw = np.zeros((nh/2,nc))
#Oiw = np.zeros((ni/2,no))
#Ohw = np.zeros((nh/2,no))
#Hw = np.zeros((ni+nc,nh))
#Cw = np.zeros((nh,nc))


#### setup agents ####
Weights = np.zeros((ni/2, nh*2+nc+no*2))
WEIGHTS = np.zeros((ni/2, nh*2+nc+no*2,nagent))
HW = np.zeros((ni+nc,nh,nagent))
CW = np.zeros((nh,nc,nagent))
OW = np.zeros((ni+nh,no,nagent))
for i in range(nagent):
	Hiw = Weights[:,0:nh]
	Hcw = Weights[:,nh:2*nh]
	Chw = Weights[:,2*nh:3*nh]
	Oiw = Weights[:,3*nh:3*nh+no]
	Ohw = Weights[:,3*nh+no:4*nh]

	HW[:,:,i] = np.vstack((Hiw,Hiw,Hcw,Hcw)) #shape (ni+nc,nh)
	CW[:,:,i] = np.vstack((Chw,Chw)) #shape (nh,nc)
	OW[:,:,i] = np.vstack((Oiw,Oiw,Ohw,Ohw)) #shape (ni+nh,no)

#I[0:6] = [0,1,2,0,0,0]


def agentcalc(loc,obj, C, Hw, Cw, Ow):
	x = np.zeros((2,2))
	Im = np.zeros((2,ni/2))
	x[:,0] = loc[0:2] + [size*np.cos(loc[2]+np.pi*.25),size*np.sin(loc[2]+np.pi*.25)]
	x[:,1] = loc[0:2] + [size*np.cos(loc[2]-np.pi*.25),size*np.sin(loc[2]-np.pi*.25)]
	
	for j in range(2):
		dif = obj[:,0:2]-x[:,j]
		d = (dif[:,0]**2 + dif[:,1]**2)**.5
		s = (1+d)**-1*Smax*(1-d*dmax**-1)
		for k in range(len(s)):
			if s[k] < 0:
				s[k] = 0
		S = np.dot(s,obj[:,2:])	
		
		Im[j,:] = sigmoid(S)
	I = np.hstack((Im[0,:],Im[1,:]))
	
	IC = np.hstack((I,C))	
	H = sigmoid((np.dot(IC,Hw)+b))	
	Cnew = sigmoid((np.dot(H,Cw)+b))
	IH = np.hstack((I,H))
	O = sigmoid((np.dot(IH,Ow)+b))	
	return Cnew, O

def predcalc(predloc,loc): # predloc = location of 1 predator, loc = location of all agents
	x = np.zeros((2,2))
	Ip = np.zeros(2)
	x[:,0] = predloc[0:2] + [size*np.cos(predloc[2]+np.pi*.25),size*np.sin(predloc[2]+np.pi*.25)]
	x[:,1] = predloc[0:2] + [size*np.cos(predloc[2]-np.pi*.25),size*np.sin(predloc[2]-np.pi*.25)]
		
	for k in range(2):
		dif = loc[:,0:2]-x[:,k]
		d = (dif[:,0]**2 + dif[:,1]**2)**.5
		s = (1+d)**-1*Smax*(1-d*dmax**-1)
		for L in range(len(s)):
			if s[L] < 0:
				s[L] = 0
		Ip[k] = sigmoid(np.array(sum(s)))
	Op = sigmoid((np.dot(Ip,Owp)+b))	
	return Op

def force(loc,mov,O):
	locnew = np.zeros(3)
	movnew = np.zeros(3)
	Frot = O[0]-O[1]
	Ftra = O[0]+O[1]-np.absolute(Frot)
	
	movnew[0:2] = mov[0:2]*(1-wtra)+(Ftra*M**-1*dt)*np.array([np.cos(loc[2]),np.sin(loc[2])])
	locnew[0:2] = loc[0:2] + 0.5*(mov[0:2]+movnew[0:2])*dt
	
	
	movnew[2] = mov[2]*(1-wrot) + size*Frot*dt*.5*inert**-1
	locnew[2] = loc[2] + .5*(mov[2]+movnew[2])*dt
	
	return locnew, movnew


OW[0:6,0,0] = [0,1,1,0,1,1]


obj = np.zeros((nfood+npred+nagent,2+nscent))

loc = np.multiply([400,400,2*np.pi],np.random.random((nagent,3)))+[-200,-200,-np.pi]

print "loc =", loc
locp = np.multiply([400,400,2*np.pi],np.random.random((npred,3)))+[-200,-200,-np.pi]
#loc = np.array([[0.,0.,0.],[100.,0.,np.pi/2]])
#mov = np.array([[0.,0.,0.],[0.,0.,0.]])

LOC = np.zeros((nagent,3,timesteps))
MOV = np.zeros((nagent,3,timesteps))
LOCP = np.zeros((npred,3,timesteps))
MOVP = np.zeros((npred,3,timesteps))

LOC[:,:,0] = loc
LOCP[:,:,0] = locp
#MOVP[:,:,0] = movp
#MOV[:,:,0] = mov



Otot = np.zeros((nagent,3))
objagent = np.hstack((loc[:,0:2],Otot))

obj = np.zeros((nfood+npred+nagent,2+ni/2))
obj[0:nfood,0:2] = 400*np.random.random((nfood,2))-200
obj[nfood:nfood+npred,0:2] = locp[:,0:2]
obj[nfood+npred:nfood+npred+nagent,0:2] = loc[:,0:2]

for i in range(nfood+npred+nagent):
	if i<nfood:
		obj[i,2:5] = [.5,1.,0.]
	else:
		if i<(nfood+npred):
			obj[i,2:5] = [1.,.5,0.]

#obj = np.array([[0.,20.,1.,0.5,0.],[40.,0.,1.,0.5,0.],[20.,0.,0.5,1.,0.],[0.,40.,0.5,1.,0.],objagent[0,:], objagent[1,:]])

print obj

OBJ = np.zeros((nfood+npred+nagent,2+ni/2,timesteps))
OBJ[:,:,0] = obj

#### setup predator ####
Ip = np.zeros(2)
Op = np.zeros(2)

Owp = np.array([[-0.1,3],[3,-0.1]])

predloc = np.hstack((obj[nfood:nfood+npred,0:2],np.zeros((npred,1))))
#predloc = np.zeros((npred,3))
predmov = np.zeros((npred,3))

LOCP[:,:,0] = predloc
MOVP[:,:,0] = predmov

OO = np.zeros((nagent,no))
CC = np.zeros((nagent,nc))

upp = 200
low = -200



###start simulation###
for i in range(timesteps-1):
	obj = OBJ[:,:,i]
	
	objmirror = np.array(obj)
	obj = np.vstack((obj,objmirror-[upp-low,0,0,0,0],objmirror+[upp-low,0,0,0,0],objmirror-[0,upp-low,0,0,0],objmirror+[0,upp-low,0,0,0],objmirror+[upp-low,upp-low,0,0,0],objmirror+[upp-low,low-upp,0,0,0],objmirror+[low-upp,low-upp,0,0,0], objmirror+[low-upp,upp-low,0,0,0]))	
	
	for j in range(nagent):
		objj = np.delete(obj, [j,2*j,3*j,4*j,5*j,6*j,7*j,8*j,9*j], axis = 0)
		CC[j,:], OO[j,:] = agentcalc(LOC[j,:,i],objj,CC[j,:],HW[:,:,j],CW[:,:,j],OW[:,:,j])
		locnew, MOV[j,:,i+1] = force(LOC[j,:,i],MOV[j,:,i],OO[j,:])
		if locnew[0]>upp:
			 locnew[0] = low
		if locnew[0]<low:
			locnew[0] = upp
		if locnew[1]>upp:
			locnew[1] = low
		if locnew[1]<low:
			locnew[1] = upp
		LOC[j,:,i+1] = locnew
	for j in range(nagent-1):
		dif = LOC[j+1:,0:2,i+1]-LOC[j,0:2,i+1]
		rsq = (dif[:,0]**2+dif[:,1]**2)
		for k in range(nagent-1-j):
			if rsq[k] < 400:
				change = -1*dif[k,:]*(rsq[k]**.5)*(2*size)**-1
				LOC[j,0:2,i+1] += change

		locmirror = np.array(loc)
	
	locmirror = np.array(LOC[:,:,i])
	loc = np.vstack((locmirror,locmirror-[upp-low,0,0],locmirror+[upp-low,0,0],locmirror-[0,upp-low,0],locmirror+[0,upp-low,0],locmirror+[upp-low,upp-low,0],locmirror+[upp-low,low-upp,0],locmirror+[low-upp,low-upp,0], locmirror+[low-upp,upp-low,0]))

	for j in range(npred):
		OOp = predcalc(LOCP[j,:,i],loc)
		locpnew, MOVP[j,:,i+1] = force(LOCP[j,:,i],MOVP[j,:,i], OOp)
		if locpnew[0]>upp:
			 locpnew[0] = low
		if locpnew[0]<low:
			locpnew[0] = upp
		if locpnew[1]>upp:
			locpnew[1] = low
		if locpnew[1]<low:
			locpnew[1] = upp
				
		LOCP[j,:,i+1] = locpnew
	for j in range(npred-1):
		dif = LOCP[j+1:,0:2,i+1]-LOCP[j,0:2,i+1]
		rsq = (dif[:,0]**2+dif[:,1]**2)
		for k in range(npred-1-j):
			if rsq[k] < 400:
				change = -1*dif[k,:]*(rsq[k]**.5)*(2*size)**-1
				LOCP[j,0:2,i+1] += change
		


print "location predator start", LOCP[:,:,0]
print "location predator +1", LOCP[:,:,1]

print "location agent start", LOC[:,:,0]
print "location agent +1", LOC[:,:,1]
print "speed agent +1", MOV[:,:,1]


plt.figure()
colorlist = ['blue', 'purple','yellow','orange','cyan','b','b','b','b','b']
for i in range(nagent):
	plt.plot(LOC[i,0,:],LOC[i,1,:],color = colorlist[i])
for i in range(npred):
	plt.plot(LOCP[i,0,:],LOCP[i,1,:],color = 'red')
plt.scatter(OBJ[0:nfood,0,0],OBJ[0:nfood,1,0],color = 'green')
plt.ylim([low,upp])
plt.xlim([low,upp])
plt.show()
