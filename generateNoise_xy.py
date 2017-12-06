# script to generate 2D noise (pink, brown, etc.)
# generates white noise movies with limited spatial and temporal
#%% frequency, via inverse fourier transform
#%% typical parameters
#%% spatfreq = .012
#%% tempfreq = 5
#%% contrast_sigma = .75
#%% duratio = 5 (mins)

# Also, generate sum of gaussians and write analytical formula
# Plus, generate analytical formula for 2d velocity field

# imports
import numpy as np
import numpy.random as npr
import numpy.fft as npf
import scipy.interpolate as spin
from numpy import linspace
import matplotlib.pyplot as plt
import meshutils
from nektarutils import mystruct, round2digit

'''
Define auxiliary functions
'''

def first(ax,ay):
	ax = ax
	ay = ay
	return ax,ay
		
def second(ax,ay):
	ax = ax
	ay = -ay
	return ax,ay

def third(ax,ay):
	ax = -ax
	ay = ay
	return ax,ay

def fourth(ax,ay):
	ax = -ax
	ay = -ay
	return ax,ay

def fifth(ax,ay):
	ax = 0
	ay = ay*np.sign(np.random.randn())
	return ax,ay

def sixth(ax,ay):
	ax = ax*np.sign(np.random.randn())
	ay = 0
	return ax,ay

options = {0: first, 1: second, 2: third, 3: fourth, 4: fifth, 5: sixth}

'''
random noise class
'''
class noisyS:
	def __init__(self, imsize = 64, maxSF=8, alpha=-1, contrastSigma=.3, ds = 3, solverdir = '/home/user'):
		self.imsize = imsize
		self.maxSF = maxSF
		self.alpha = alpha
		self.cSigma = contrastSigma
		self.data = np.zeros((imsize,imsize))
		self.ds = ds 
		self.solverdir = solverdir
		np.random.seed()

	def generate_noise(self):
		#%% stimulus/display parameters
		halfsize = int(self.imsize/2)
		degperpix = self.ds/self.imsize #2 is the domain size

		nyq_pix = 0.5
		nyq_deg=nyq_pix/degperpix #=self.imsize/4 in this case
		freqInt_pix = nyq_pix / (0.5*self.imsize)

		#% cutoffs in terms of frequency intervals
		maxFreq_pix = self.maxSF*degperpix
		spatCutoff = round(maxFreq_pix / freqInt_pix) # obs: this is basically self.maxSf/nyq_pix or something like that


		#% generate frequency spectrum (invFFT)
		offset=3
		range_mult =1
		#for noise that extends past cutoff parameter (i.e. if cutoff = 1sigma)
		#range_mult=2;
		sco = int(range_mult*spatCutoff)
		spaceRange = [int(self.imsize/2 - range_mult*spatCutoff), int(self.imsize/2 + range_mult*spatCutoff+1)]
		#print(spaceRange)  
		#np.arange(1 + self.imsize/2 - range_mult*spatCutoff, self.imsize/2 + range_mult*spatCutoff+2, dtype= np.int16)
		sizeRng = spaceRange[1]-spaceRange[0]
		#print(spaceRange.shape)
        # can put any other function to describe frequency spectrum in here,
        # e.g. gaussian spectrum
		xr = np.arange(-range_mult*spatCutoff,range_mult*spatCutoff+1)
		yr = np.arange(-range_mult*spatCutoff,range_mult*spatCutoff+1)
		#print(xr)
		[x, y] = np.meshgrid(xr,yr)
		xr = None
		yr = None 
		# use = exp(-1*((0.5*x.^2/spatCutoff^2) + (0.5*y.^2/spatCutoff^2)).astype('int32')
		# use = (((x.^2 + y.^2)<=(spatCutoff^2))).astype('int32')
		use = (((x**2 + y**2)<=(spatCutoff**2)))*(np.sqrt(x**2 + y**2 +offset)**self.alpha).astype('float32')

		x = None
		y = None

		#%% 
		invFFT = np.zeros((self.imsize,self.imsize),dtype=np.complex64)
		mu = np.zeros((sizeRng,sizeRng))
		sig = np.ones((sizeRng,sizeRng))
		
		invFFT[spaceRange[0]:spaceRange[1], spaceRange[0]:spaceRange[1]] = use * npr.randn(sizeRng,sizeRng) * np.exp(2*np.pi*1j*npr.rand(sizeRng, sizeRng))
		use = None

		#% in order to get real values for image, need to make spectrum symmetric
		fullspace = [halfsize-sco , halfsize+sco+1]
		backspace = [halfsize-sco-1, halfsize+sco]

		tmp = invFFT[backspace[1]:backspace[0]:-1][:, halfsize-1:backspace[0]:-1]
		#print(fullspace,backspace)
		invFFT[fullspace[0]:fullspace[1] , halfsize+1:fullspace[1]] = tmp.conj()

		invFFT[halfsize+1:fullspace[1], halfsize] = invFFT[halfsize-1:backspace[0]:-1, halfsize].conj()

		invFFT = npf.ifftshift(invFFT)

		### invert FFT

		imraw = npf.ifftn(invFFT).real
		#print(np.imag(npf.ifftn(invFFT)).max(),np.real(npf.ifftn(invFFT)).max())
		immean = imraw.mean()
		immax = imraw.std()/self.cSigma
		immin = -1*immax
		# here is normalised. There is going to be a certain amount of pixels that are outside of 0 to 1
		# range: those are the ones that needs to be saturated.
		imraw = (imraw - immin-immean) / (immax - immin)
		#data = (np.floor(imraw*255)+1).astype(np.uint8)
		
		data = np.clip(imraw,0,1).astype(np.float32)
		imraw = None
		
		fig, ax = plt.subplots(nrows = 1, ncols = 1)
		ax.matshow(data)
		fig.savefig('pinknoise.png')   # save the figure to file
		plt.close(fig)
		return data
	
	def pad_domain(self,newds,mymode='constant',c=0):
		# for now, just pad everything with 0
		newsize = np.ceil(self.imsize*newds/self.ds)
		newsize = newsize if (newsize-self.imsize)%2==0 else newsize-1
		sizediff = int((newsize - self.imsize)/2)
		newdata = np.pad(self.data,((sizediff,sizediff),(sizediff,sizediff)),mode = mymode)
		#fig, ax = plt.subplots(nrows = 1, ncols = 1)
		#ax.matshow(newdata)
		#fig.savefig('newpinknoise.png')   # save the figure to file
		#plt.close(fig)
		return (newdata,int(newsize))
		
	def interpolate_noise(self, mode = 'size', newsize = (128,128), pts = -1, extra = None):
		# untested routine. There might be bugs
		if pts == -1:
			# don't use points
			mode = 'size'
		elif pts == -2 & extra:
			# load points from text file
			mode = 'points'
			xnew = None
			ynew = None
		elif pts == -3 & extra:
			#load points from pickle file
			mode = 'points'
			xnew = None
			ynew = None
		elif pts == -4 & extra:
			#load points from mat file
			mode = 'points'
			xnew = None
			ynew = None
		
		oldsize = self.data.shape
		dx = self.ds/oldsize[1]/2 #this is the coordinate of a half pixel
		dy = self.ds/oldsize[0]/2
		# define x,y grid outside of -1 and 1, and pad image with one row/column of zeros at
		# the edges to have the function fully defined on the whole [-1,1]^2 domain
		xg = linspace(-self.ds/2-dx, self.ds/2+dx, oldsize[0]+2)
		#print(xg.shape)
		yg = linspace(-self.ds/2-dy, self.ds/2+dy, oldsize[1]+2)
		data = np.pad(self.data,((1,1),(1,1)),'constant')
		interpolator = spin.RectBivariateSpline(xg,yg,data)
		# now interpolate
		if mode == 'size':
			dxnew = self.ds/newsize[1]/2
			dynew = self.ds/newsize[0]/2
			xnew = linspace(-self.ds/2+dx, self.ds/2-dx, newsize[0])
			ynew = linspace(-self.ds/2+dy, self.ds/2-dy, newsize[1])
			return (interpolator(xnew,ynew),xnew,ynew)
		else:
			return (interpolator.ev(xnew,ynew),xnew,ynew)
			
'''
Sum of Gaussians generator
'''
def generate_gaussians(Ngs,pr):
	U0 = mystruct()
	U0.Ngs = Ngs
	U0.x0 = np.zeros(Ngs)
	U0.y0 = np.zeros(Ngs)
	U0.sigma = np.zeros(Ngs)
	U0.gamma = np.zeros(Ngs)
	U0.A = np.zeros(Ngs)
	U0.theta = np.zeros(Ngs)
	init_sol = '\t\t<E VAR="u" VALUE= "A0*exp(-(x^2 + y^2)/(2*sigma0^2))/sigma0 '
	Nsr = 0
	for ii in range(Ngs):
		flag = 1;
		if ii>0 and flag:
			U0.x0[ii] = round2digit(pr.minx0 + pr.rx0*np.random.rand(),3)
			U0.y0[ii] = round2digit(pr.miny0 + pr.ry0*np.random.rand(),3)
			U0.theta[ii] = round2digit(pr.mint + pr.rt*np.random.rand(),3)
			U0.gamma[ii] = round2digit(pr.ming + pr.rg*np.random.rand(),3)
			U0.sigma[ii] = round2digit(pr.mins + pr.rs*np.random.rand(),3)
			U0.A[ii] = round2digit((pr.meanA + pr.rA*np.random.rand())*np.sign(np.random.rand()),3)
			
			if min(abs(U0.x0[ii]-U0.x0[0:ii]))>.1 and min(abs(U0.y0[ii]-U0.y0[0:ii]))>.1 and max(U0.sigma)>.25:
				flag = 0
			else:
				Nsr += 1
		else:
			U0.x0[ii] = round2digit(pr.minx0 + pr.rx0*np.random.rand(),3)
			U0.y0[ii] = round2digit(pr.miny0 + pr.ry0*np.random.rand(),3)
			U0.theta[ii] = round2digit(pr.mint + pr.rt*np.random.rand(),3)
			U0.gamma[ii] = round2digit(pr.ming + pr.rg*np.random.rand(),3)
			U0.sigma[ii] = round2digit(pr.mins + pr.rs*np.random.rand(),3)
			U0.A[ii] = round2digit((pr.meanA + pr.rA*np.random.rand())*np.sign(np.random.rand()),3)
		init_sol = init_sol + ' + ' + '({})*exp(-(((x*cos({}) - sin({})*y - ({}))/{})^2)/2 -{}*(((y*cos({}) + sin({})*x - ({}))/{})^2)/2)'.format(U0.A[ii],U0.theta[ii],U0.theta[ii],U0.x0[ii],U0.sigma[ii],U0.gamma[ii],U0.theta[ii],U0.theta[ii],U0.y0[ii],U0.sigma[ii])
		#
	print('Generating gaussians: samples rejected: ' + str(Nsr) + '\n')
	return (U0,init_sol)	

'''
Velocity field generator
'''		
def generate_vel(idv,pr):
	if idv == 0:
		# spatially uniform field
		ax = round2digit(pr.min_ax[idv] + (pr.max_ax[idv]-pr.min_ax[idv])*np.random.beta(1.5,1.5),3)
		ay = round2digit(pr.min_ax[idv] + (pr.max_ax[idv]-pr.min_ax[idv])*np.random.beta(1.5,1.5),3)
		cx = 3000
		cy = 3000
		# decide what type it's going to be: (+,+),(+,-),(-,+),(-,-),(0,+),(0,-),(+,0),(-,0)
		case = np.random.randint(low = 0, high = 6)
		ax,ay = options[case](ax,ay)
		Vx= '\t\t<E 	VAR="Vx" VALUE="ax" 	/> \n'
		Vy = '\t\t<E 	VAR="Vy" VALUE="ay" 	/> \n'
		Vsave = 'spatially_uniform'

	if idv == 1:
		# constant angular velocity
		ax = round2digit(pr.min_ax[idv] + (pr.max_ax[idv]-pr.min_ax[idv])*np.random.beta(1.5,1.5),3)
		ay = round2digit(pr.min_ax[idv] + (pr.max_ax[idv]-pr.min_ax[idv])*np.random.beta(1.5,1.5),3)
		cx = round2digit(pr.min_ac + (pr.max_ac-pr.min_ac)*np.random.beta(1.5,1.5),4)
		cy = round2digit(pr.min_ac + (pr.max_ac-pr.min_ac)*np.random.beta(1.5,1.5),4)
		Vx = '\t\t<E 	VAR="Vx" VALUE="-ax*(y - cx)" 	/> \n'
		Vy = '\t\t<E 	VAR="Vy" VALUE="ay*(x - cy)" 	/> \n'
		Vsave = 'constant_angular'

	if idv == 2:
		# constant linear velocity
		ax = round2digit(pr.min_ax[idv] + (pr.max_ax[idv]-pr.min_ax[idv])*np.random.beta(1.5,1.5),3)
		ay = round2digit(pr.min_ax[idv] + (pr.max_ax[idv]-pr.min_ax[idv])*np.random.beta(1.5,1.5),3)
		cx = round2digit(pr.min_ac + (pr.max_ac-pr.min_ac)*np.random.beta(1.5,1.5),4)
		cy = round2digit(pr.min_ac + (pr.max_ac-pr.min_ac)*np.random.beta(1.5,1.5),4)
		Vx = '\t\t<E 	VAR="Vx" VALUE="-ax*(y - cx)/sqrt(x^2 + y^2 + beta)"	 /> \n'
		Vy = '\t\t<E 	VAR="Vy" VALUE="ay*(x - cy)/sqrt(x^2 + y^2 + beta)" 	/> \n'
		Vsave = 'constant_linear'		

	return Vx,Vy,Vsave,ax,ay,cx,cy

#p = noisyS()
#p.data = p.generate_noise()
#p.newdata = p.interpolate_noise()[0]
#print(p.newdata.shape)
