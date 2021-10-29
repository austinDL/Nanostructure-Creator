#IMPORTS
import numpy as np
from numpy import array, pi, cos, sin, tan
from numba import typeof, njit, int64, float64, boolean
from numba.experimental import jitclass


#### Functions for Numba operations ####

@njit
def linspace(start,stop,num):
    return np.linspace(start,stop,num)

@njit
def dot(ar1,ar2):
    return ar1[0]*ar2[0] + ar1[1]*ar2[1] + ar1[2]*ar2[2]

#### Functions for Numba operations ####


#### Light Ray Class ####

'''spec = [
    ('pos', float64[:,:]),
    ('c', float64[:,:]),
    ('d', float64[:])
]

@jitclass(spec)'''
class Ray:
  def __init__(self, maxVal=0.1, arrayNum=36):
    self.init_rayParams(maxVal, arrayNum)
    

  def init_rayParams(self, maxVal, arrayNum):
    ### INITIALIZE THE PARAMETERS OF THE RAYS ###

    range_ = linspace(-maxVal,maxVal,arrayNum)

    self.pos = np.zeros((arrayNum**2, 3))           # Position of Rays
    self.c   = np.zeros((arrayNum**2, 3))           # Photon Velocity Vector
    self.d   = np.zeros(arrayNum**2)                # Distance Travelled by the Photons

    for k in range(arrayNum):
      for i in range(arrayNum):
        #Define the index to convert to a 2D array
        ind = i + k*arrayNum

        self.pos[ind] = array([range_[i], 0, range_[k]])
        self.c[ind]   = array([0,1,0])


  def changeC(self,n, ind):
    ### CHANGE THE DIRECTION OF THE LIGHT RAY AFTER REFLECTING OFF OF MIRROR###
    
    self.c[ind] -= 2*dot(self.c[ind],n) * n
        
  def move2plane(self,pContact, ind):
    ### MOVE THE LIGHT RAY TO A DESIGNATED PLANE ###
    
    pathLen = pContact - self.pos[ind]
    
    #Increase the distance travelled by the light
    self.d[ind] += (dot(pathLen,pathLen))**0.5
    
    #Move the light ray to the point of contact on the mirror
    self.pos[ind] = pContact
        
  def move2diffractionZone(self, ind):
    ### MOVE THE LIGHT RAY TO THE DIFFRACTION ZONE (which is a plane at y = 1 and n = [0,1,0]) ###
    
    #The diffraction zone is a plane at y = 1
    #Define the features of the diffraction zone plane
    n = array([0,1,0])
    pos = array([0,1,0])
    
    #If the Ray isn't travelling in the y-direction, then it doesn't reach the diffraction zone
    if self.c[ind,1] != 0:
      
      #Find the time taken to get to the diffraction zone
      t = dot(n,pos - self.pos[ind]) / dot(n,self.c[ind])

      #Determine the point of contact
      pContact = self.pos[ind] + self.c[ind]*t
      
      #Move to the diffraction zone
      self.move2plane(pContact, ind)

    else:
      #Set the x-z component to infinity, since the Ray doesn't reach the diffraction zone
      self.pos[ind,::2] = np.float(np.inf)

#### Light Ray Class ####
    

#### Mirror Class ####

'''spec = [
  ('pos', float64[:,:]),
  ('n', float64[:,:])
]

@jitclass(spec)'''
class Mirror:
  def __init__(self, maxVal=0.1, arrayNum=36):
    self.init_mirrorParams(maxVal, arrayNum)
        

  def init_mirrorParams(self, maxVal, arrayNum):
  ### SET THE INITIAL CONDITIONS OF THE MIRROR BEFORE AGENT'S INTERACTIONS ###

    range_ = linspace(-maxVal, maxVal, num=arrayNum)

    self.pos = np.zeros((arrayNum**2, 3))         # Position of Mirrors
    self.n = np.zeros((arrayNum**2, 3))           # Mirror Normal Vector
    
    for k in range(arrayNum):
      for i in range(arrayNum):
        #Define the index to convert to a 2D array
        ind = i + k*arrayNum
        
        self.pos[ind,:] = array([range_[i], 0.5, range_[k]])
        self.n[ind,:] = array([0,1,0])

    
  def setN(self,theta,phi,ind):
    ### AGENT SETS THE NORMAL VECTOR OF THE MIRROR ###
    
    self.n[ind,:] = np.array([sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)])
        
  def setYpos(self,Y, ind):
    ### AGENT SETS THE POSITION OF THE MIRROR
    
    self.pos[ind,1] = Y
        
  def contact(self, Rays, ind):
    ### REFLECT THE LIGHT RAYS OFF OF A MIRROR ###
    
    #Determine which photon-mirror pairs interact with eachother
    if 0 < self.pos[ind,1] < 1:
      #Move the Ray to the diffraction zone
      Rays.move2plane(self.pos[ind], ind)
      Rays.changeC(self.n[ind], ind)
      Rays.move2diffractionZone(ind)
        
    else:
      Rays.pos[ind,1] = 1
      Rays.d[ind] += 1

#### Mirror Class ####
    

#### Interference Camera Class ####

'''spec = [
  ('maxVal', float64),
  ('arrayNum', int64),
  ('pixelSize', float64),
  ('img', float64[:,:]),
  ('ray_pos', float64[:,:]),
  ('ray_d', float64[:])
]

@jitclass(spec)'''
class Interference_Camera:
  def __init__(self, maxVal=0.1, arrayNum=36):
    self.maxVal = maxVal
    self.arrayNum = arrayNum
    self.pixelSize = 2*maxVal/(arrayNum - 1)
    self.img = np.zeros((arrayNum, arrayNum)) #Initially, there is no interference

  def get_pixel_ind(self, ray_pos, ray_c):
    ### DETERMINE WHERE THE RAY IS IN THE IMAGE ###

    #Make sure the Ray has reached the diffraction zone
    if (-self.pixelSize < ray_pos[0] < 2*self.maxVal+self.pixelSize) and (-self.pixelSize < ray_pos[1] < 2*self.maxVal+self.pixelSize) and (ray_c[1] > 0):
      ray_pixel_ind = np.floor(np.abs(ray_pos)/self.pixelSize)

    #If the Ray is outside the diffraction zone, then return a nonsense pixel ind
    else:
      ray_pixel_ind = np.array([-2,-2])

    return ray_pixel_ind
      

  def set_pixel_vals(self, Rays, ind, ref_IMG):
    #Save the important paramaters of the Rays
    self.ray_pos = Rays.pos[:,::2] + self.maxVal #The positions are shifted to match pixel positions: [0,0]->[2*maxVal, 2*maxVal]
    self.ray_d   = Rays.d
    #Initialize reward parameters
    reward, reward_norm = 0,0
    
    ray_pixel_ind = self.get_pixel_ind(self.ray_pos[ind], Rays.c[ind])

    #Ensure the Ray is inside the diffraction zone
    if ray_pixel_ind[0] >= 0:
      #Determine if we are on the edge of the diffraction zone
      x_edge = True if ray_pixel_ind[0]==self.arrayNum-1 else False
      z_edge = True if ray_pixel_ind[1]==self.arrayNum-1 else False

      #Loop through neighboring pixels
      for x_shift in range(2-x_edge):
        for z_shift in range(2-z_edge):
          pixel_ind = ray_pixel_ind + np.array([x_shift, z_shift])
          pixel_pos = pixel_ind*self.pixelSize

          #Set the interference values in the nanostructure image
          image_ind = (np.int(pixel_ind[1]), np.int(pixel_ind[0]))
          self.img[image_ind] = self.get_interference(pixel_pos)

          #Determine how well the actions influencing this photon meet the requirements of the target nanostructure
          if self.img[image_ind]!= 0: #0 means no interaction, so we want to penalize these actions
            reward += np.abs(self.img[image_ind] - ref_IMG[image_ind])/2
            reward_norm += 1

    return reward/reward_norm if reward_norm>0 else 0


  def get_interference(self, pixel_pos, tol=0.9):
    #Initialize
    pixel_val = 0
    pixel_norm = 0
    tol *= self.pixelSize

    #Find the ray positions that interfere within the pixel
    pos_diff = np.abs(pixel_pos - self.ray_pos)
    interaction_ind = np.where((pos_diff[:,0] <= tol) & (pos_diff[:,1] <= tol))
    inter_pos, inter_d = self.ray_pos[interaction_ind], self.ray_d[interaction_ind]
    
    #Weighted average of the light interferences in a given pixel
    for i in range(len(interaction_ind[0]-1)):
      for neighbor_pos, neighbor_d in zip(inter_pos[i+1:], inter_d[i+1:]):
        A = self.get_interference_area(neighbor_pos, inter_pos[i], pixel_pos)
        I = self.get_interference_intensity(neighbor_d, inter_d[i])

        pixel_val  += I*A
        pixel_norm += A

    return pixel_val / pixel_norm if pixel_norm > 0 else 0

  def get_interference_area(self, neighbor_pos, ref_pos, pixel_pos):
    ### DETERMINES THE AREA OF INTERSECTION BETWEEN THE TWO PHOTONS AND THE PIXEL BIN ###
    Ax = min(neighbor_pos[0], ref_pos[0], pixel_pos[0]) + self.pixelSize - max(neighbor_pos[0], ref_pos[0], pixel_pos[0])
    Az = min(neighbor_pos[1], ref_pos[1], pixel_pos[1]) + self.pixelSize - max(neighbor_pos[1], ref_pos[1], pixel_pos[1])

    return Ax*Az if Ax>0 and Az>0 else 0

    
  def get_interference_intensity(self, neighbor_d, ref_d, lambda_Rays=0.01):
    return cos(np.abs(ref_d - neighbor_d) * 2*pi/lambda_Rays)

#### Interference Camera Class ####


#### RL Environment Class ####

'''spec = [
  ('maxVal', float64),
  ('arrayNum', int64),
  ('ref_IMG', float64[:,:]),
  ('state', int64[:]),
  ('Rays', typeof(Ray())),
  ('Mirrors', typeof(Mirror())),
  ('used_mirrors', boolean[:]),
  ('Camera', typeof(Interference_Camera())),
  ('action_max', float64[:]),
  ('action_min', float64[:]),
  ('reward_power', float64),
  ('old_scale', float64),
  ('TF', boolean)
]

@jitclass(spec)'''
class Environment:

  def __init__(self, maxVal=0.1, arrayNum=36, 
               a_max=array([ 1,  pi/4, .75*pi]), a_min=array([-1, -pi/4, .25*pi ]), TF=True, 
               ref_IMG=np.zeros([36,36]), power=1, old_scale=0):

    self.maxVal = maxVal
    self.arrayNum = arrayNum
    self.ref_IMG = ref_IMG

    #Determine which transition function we are using
    self.TF = TF
    
    if self.TF:
      self.state = np.random.choice(self.arrayNum, 2)
    else:
      self.state = np.array([0,0])
    
    #Define the range of values in the action space
    self.action_max = a_max
    self.action_min = a_min
    
    #Initialize the Optical Equipment
    self.Rays    = Ray(maxVal, arrayNum)
    self.Mirrors = Mirror(maxVal, arrayNum)
    self.used_mirrors = np.zeros(arrayNum**2)
    self.Camera  = Interference_Camera(maxVal, arrayNum)

    #Reward Parameters
    self.reward_power = power
    self.old_scale = old_scale
    
    
  def change_state(self):
    ### Transition Function ###
    #This transition function changes the state the same way, regardless of the action

    self.state[0] += 1
    
    if self.state[0] == self.arrayNum:
      self.state[0] = 0
      self.state[1] += 1
    
    #Handle the terminal state
    if self.state[1] == self.arrayNum:
      self.state = np.array([self.arrayNum-1, self.arrayNum-1])
    
    return self.state
    
  def change_state_TF(self, last_ind):
    ### Transition Function ###
    #This transition function transitions to the state where the interaction occured
      
    #Check to see if we are in the terminal state
    if np.sum(1-self.used_mirrors) == 0:
      self.state[0] = last_ind % self.arrayNum
      self.state[1] = last_ind // self.arrayNum
      #Determine the next state
      next_state = self.state

    else: 
      #Determine the index of mirror that will be used next
      contact_pos = self.Rays.pos[last_ind]

      #Shift the position of the used mirrors so that they can't be the next state
      shift = 100_000
      self.Mirrors.pos[self.used_mirrors] += shift

      pos_diff = (self.Mirrors.pos[:,0] - contact_pos[0])**2 + (self.Mirrors.pos[:,2] - contact_pos[2])**2

      #Return back to its original state
      self.Mirrors.pos[self.Mirrors.used] -= shift

      next_ind = np.argmin(pos_diff)
      
      #Determine the next state
      self.state[0] = next_ind % self.arrayNum
      self.state[1] = next_ind // self.arrayNum
      next_state = self.state
        
    return next_state
      
      
    
  def step(self, action):
      
    #Initialize the state index
    ind = np.int(self.state[0] + self.state[1]*self.arrayNum)

    done = False
    #Deconstruct the action into its components
    actions = self.translate(action)
    y_pos, theta, phi = actions
    
    #Agent acts on the Environment by setting the orientation of the mirror
    self.Mirrors.setN(theta, phi, ind)
    self.Mirrors.setYpos(y_pos, ind)
  
    #Model the interactions
    self.Mirrors.contact(self.Rays, ind)
    reward = self.Camera.set_pixel_vals(self.Rays, ind, self.ref_IMG)
    
    #Remove the used Mirror
    self.used_mirrors[ind] = True
  
    #Transition to the next state
    current_state = self.state.copy()
    if self.TF:
      next_state = self.change_state_TF(last_ind=ind)
    else:
      next_state = self.change_state()

    #If we land back in the same state, then we are in the terminal state
    if (next_state[0]==current_state[0]) and (next_state[1]==current_state[1]):
      done = True
      reward = 0
    
    return next_state, reward, done
    
    
  def translate(self, action):
    #Translate the action from (-1,1) to the range of the actions
    actions = (action+1)/2 * (self.action_max - self.action_min) + self.action_min
    return actions
    
    
  def reset(self):
    #Reset the state
    if self.TF:
      self.state = np.random.choice(self.arrayNum, 2)
    else:
      self.state = np.array([0,0])
    
    #Reset the Optics Apparatus
    self.Rays = Ray(self.maxVal, self.arrayNum)
    self.Mirrors = Mirror(self.maxVal, self.arrayNum)
    self.used_mirrors = np.zeros(self.arrayNum**2)
    self.Camera = Interference_Camera(self.maxVal, self.arrayNum)
    
    state, done = self.state, False
    return state, done
    
    
  def render(self):
    return self.Camera.img

#### RL Environment Class ####
    