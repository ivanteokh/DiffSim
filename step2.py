import numpy as np
import scipy.stats as stats
import scipy.spatial as spatial
import scipy.interpolate as interp
import scipy.integrate as integr
import random

Num_nodes=100					# Number of nodes on grid

###### Spherically symmetric distribution #####
Distribution_sphere_center = [0,0,0]		# Set center of distribution
Dist_grid_spacing = 1.				# Set distance between adjacent distribution grid points
System_center = np.array([1,1,1])		# Set center of system
Radial_distribution = "r**2 + 2 * r + 1" 	# Set radial distribution
System_size = np.array([100.,100.,70.])		# Set system dimensions

def sym_sphere_grid(Radial_distribution,Distribution_sphere_center,System_center,System_size,Dist_grid_spacing):
    System_center = np.array(System_center)
    System_size = np.array(System_size)
    Num_grid_nodes = map(int,[x/Dist_grid_spacing for x in System_size]) 	# calculate number of grid nodes along each dimension, set to integer values
    Dist_grid = np.zeros(Num_grid_nodes)					# Initialize distribution grid
     
    # For each node on distribution grid, calculate the distribution using the radial distribution function.
    for i in range(int(Num_grid_nodes[0])):
       for j in range(int(Num_grid_nodes[1])):
            for k in range(int(Num_grid_nodes[2])):
                grid_node_position = np.array([i,j,k]) * Dist_grid_spacing - System_size/2 + System_center	# calculate position vector of node in the system
                r = np.linalg.norm(grid_node_position - Distribution_sphere_center)				# calculate distance of node from the center of distribution
                Dist_grid[i,j,k] = eval(Radial_distribution)						# calculate and assign the distribution function value
     
    return(Dist_grid)

###############################################

# arranges an input list of 3D points forming a polygon in cw/ccw order
def order_arrange_3D(points_list):
    points_list = points_list * 1.0									# convert to float
    N = len(points_list)										# number of points
    points_center = sum(points_list) / N								# centroid of points
    centered_points_list = points_list - points_center							# positions of points wrt centroid
    parallel_indicator = 0										# check if points used to obtain normal are collinear with center	
    counter = 1												# counter to move through points
     
    while (parallel_indicator == 0 and counter < N):
        normal_vec = np.cross(centered_points_list[0],centered_points_list[counter]) / np.linalg.norm(np.cross(centered_points_list[0],centered_points_list[counter]))	# unit normal of the polygon
        parallel_indicator = np.linalg.norm(np.cross(centered_points_list[0],centered_points_list[counter]))
        counter += 1
     
    V0 = centered_points_list[0]									# use first point in list as reference
    V0V1_angle=np.zeros([N,1])										# initialize vector of angles
    for i in range(1,N):
        V1 = centered_points_list[i]
        normdotprod = np.dot(V0,V1) / (np.linalg.norm(V0) * np.linalg.norm(V1))
         
        if abs(normdotprod) > 1:
            normdotprod = normdotprod / abs(normdotprod)
         
        V0V1_angle[i] = np.arccos(normdotprod)								# calculate acute angle between V0 and V1
        crossprod = np.cross(V0,V1)
         
        if np.dot(crossprod,normal_vec) < 0:								# convert acute angle to obtuse if it occurs in 3rd or 4th quadrant
            V0V1_angle[i] = 2 * np.pi - V0V1_angle[i]							# populate vector of angles
         
    points_angle_list = np.append(points_list,V0V1_angle,axis=1)					# construct array of point positions with angles
    points_angle_list = points_angle_list[np.argsort(points_angle_list[:,3])]				# sort points by angles
    return np.array(points_angle_list[:,0:3])  								# return sorted point positions 

################################################

# reads in a dx file and outputs an object containing an array containing the volumetric data and corresponding attributes
class readdx(object):
    def __init__(self,filename):
        infile = open(filename,'r')
        line1 = infile.readline()[35:].split()
        line2 = infile.readline()[6:].split()
        line3 = infile.readline()[5:].split()
        line4 = infile.readline()
        line5 = infile.readline()
        line6 = infile.readline()
        line7 = infile.readline()[45:].split()
        self.numval = int(line7[0])
        self.grid = map(int,line1)
        self.origin = map(float,line2)
        self.delta = float(line3[0])
        self.data = np.zeros(np.array(self.grid))
        p = 0
        rawdat = []
        line = infile.readline().split()
        while line[0].isdigit():
            rawdat[p:p+3] = map(float,line)
            p = p+3   
            line = infile.readline().split() 
        p=0
        self.cumsumvec = np.cumsum(np.array(rawdat))
        if np.max(self.cumsumvec) > 0:
            self.cumsumvec = self.cumsumvec / np.max(self.cumsumvec)
        else:
            self.cumsumvec = np.arange(1,1+len(self.cumsumvec),1,float)/(1+len(self.cumsumvec))
        for k in range(self.grid[2]):
            for j in range(self.grid[1]):
                for i in range(self.grid[0]):
                    self.data[i,j,k] = rawdat[p]
                    p = p+1
         

# decompose system into cubes (overlapping) so that search algorithms can cover single cubes instead of the whole system
# inputs: size of cube, set of points to be categorized by cube
class cube_sys(object):
    def __init__(self,W,cube_size):
        # assign to each cube a 1D integer index             
        def cube_index(cube_address,cube_num):
            cube_address = np.array(cube_address)
            if cube_address.shape == (3,):
                return cube_address[0] * cube_num[1] * cube_num[2] + cube_address[1] * cube_num[1] + cube_address[2]
            else:
                return cube_address[:,0] * cube_num[1] * cube_num[2] + cube_address[:,1] * cube_num[1] + cube_address[:,2]
            
        self.cube_size = cube_size
        self.W = W
             
        minW = np.min(W,axis=0)
        maxW = np.max(W,axis=0)
        sizeW = maxW - minW	# measure the size of the system
        cube_num = np.ceil(1.0 * sizeW / cube_size )    # number of cubes in each dimension
        cube_buffer = cube_size * 0.3	#  buffer zone for proximal nodes to be included in a cube
                    
        if np.prod(cube_num) == 1:
            raise Exception("What's the point of defining a cube that contains the whole set of points?")
                     
        cube_address = np.floor( (W-minW) / cube_size ) 	# integer coordinates of cube 
                     
        self.cube_of = cube_index(cube_address,cube_num) 	# assign the cube at each address with an integer index
        self.nodes_in_cube = [ np.array(range(len(self.cube_of)))[self.cube_of == i] for i in range(int(np.prod(cube_num)))] # displays nodes in each cube (including buffer nodes)
                     
        for i in range(len(self.W)):
            cube_sharing_1 = []
            cube_sharing_2 = [0,0,0]
            for j in range(3):
                if ((cube_address[i,j]+1)*cube_size - cube_buffer < self.W[i,j] - minW[j]) and (cube_address[i,j] < cube_num[j]-1):
                    cube_sharing = [0,0,0]
                    cube_sharing[j] = 1
                    cube_sharing_1.append(cube_sharing)
                    cube_sharing_2 = list(np.array(cube_sharing_2) + np.array(cube_sharing))
                elif (cube_address[i,j]*cube_size + cube_buffer > self.W[i,j] - minW[j]) and (cube_address[i,j] > 0):
                    cube_sharing = [0,0,0]
                    cube_sharing[j] = -1
                    cube_sharing_1.append(cube_sharing)
                    cube_sharing_2 = list(np.array(cube_sharing_2) + np.array(cube_sharing))
            if len(cube_sharing_1) > 0:
                for j in range(len(cube_sharing_1)):
                    cube_add = int(cube_index(cube_address[i]+cube_sharing_1[j],cube_num)) 
                    self.nodes_in_cube[cube_add] = np.append(self.nodes_in_cube[cube_add],i)        
            if len(cube_sharing_1) > 1:
                cube_add = int(cube_index(cube_address[i]+cube_sharing_2,cube_num))
                self.nodes_in_cube[cube_add] = np.append(self.nodes_in_cube[cube_add],i)
                if len(cube_sharing_1) == 3:
                    for j in range(3):
                        cube_add = int(cube_index(cube_address[i]+cube_sharing_1[j]+cube_sharing_1[(j+1)%3],cube_num))
                        self.nodes_in_cube[cube_add] = np.append(self.nodes_in_cube[cube_add],i)
 
# Create the Voronoi grid class
class Vor_Grid(object):
    # the class will be defined by the number of nodes, the center of the grid, and the grid dimensions
    def __init__(self,num_nodes_init,grid_center,grid_dim_xyz):
        self.num_nodes_init = num_nodes_init
        self.grid_center = grid_center
        self.grid_dim_xyz = grid_dim_xyz
        self.status = 0
        Node_spacing = ( np.prod(self.grid_dim_xyz) / self.num_nodes_init )**(1.0/3)	# Calculate distance between adjacent nodes.
        Num_nodes_X = np.ceil( self.grid_dim_xyz[0] / Node_spacing ) + 1 	# Calculate number of nodes along each dimension. 
        Num_nodes_Y = np.ceil( self.grid_dim_xyz[1] / Node_spacing ) + 1		 
        Num_nodes_Z = np.ceil( self.grid_dim_xyz[2] / Node_spacing ) + 1
        self.num_nodes = Num_nodes_X * Num_nodes_Y * Num_nodes_Z		# Re-calculate total number of nodes using rounded up values
        self.W = np.zeros([int(Num_nodes_X * Num_nodes_Y * Num_nodes_Z),3])	# Initialize Voronoi grid
        for k in range(int(Num_nodes_Z)):					# Set initial configuration to be Cartesian grid
            self.W[k*Num_nodes_X*Num_nodes_Y:(1+k)*Num_nodes_X*Num_nodes_Y ,2] = k * Node_spacing - self.grid_dim_xyz[2] / 2 + self.grid_center[2]
            for j in range(int(Num_nodes_Y)):
                self.W[k*Num_nodes_X*Num_nodes_Y + j*Num_nodes_X : k*Num_nodes_X*Num_nodes_Y + (j+1)*Num_nodes_X ,1] = j * Node_spacing - self.grid_dim_xyz[1] / 2 + self.grid_center[1]
                for i in range(int(Num_nodes_X)):
                    self.W[k*Num_nodes_X*Num_nodes_Y + j*Num_nodes_X + i ,0] = i * Node_spacing - self.grid_dim_xyz[0] / 2 + self.grid_center[0]
        self.status = 1
      
    def adapt(self,density_map,total_iteration_num,epsilon_i,epsilon_f,lambda_i,lambda_f):
          
        def iterate(W,iteration_num,total_iteration_num,drawn_point,epsilon_i,epsilon_f,lambda_i,lambda_f):
            displacement_from_point = drawn_point - W                          # Calculate list of displacements of drawn point from grid nodes
            distance_from_point = (displacement_from_point[:,0]**2 + displacement_from_point[:,1]**2 + displacement_from_point[:,2]**2 )**(0.5)     # Calculate distance of grid nodes from drawn point
            ranklist = stats.rankdata(distance_from_point) - 1.     # Use ranklist function from scipy.stats to create rank vector of distance list.
            epsilon_now = epsilon_i * (epsilon_f/epsilon_i)**(iteration_num/total_iteration_num)            # calculate value of epsilon for current iteration
            lambda_now = lambda_i * (lambda_f/lambda_i)**(iteration_num/total_iteration_num)                # calculate value of lambda for current iteration
            return W + epsilon_now * np.array( [ np.exp( -ranklist[i] / lambda_now ) * displacement_from_point[i,:] for i in range(len(W))] )   # update grid node positions
        for t in range(0,total_iteration_num):
            die = random.uniform(0,np.max(density_map.cumsumvec))
            drawn_point_index = np.min(np.argwhere(density_map.cumsumvec >= die))
            drawn_point = [ np.floor(drawn_point_index / (density_map.grid[1] * density_map.grid[2])) , np.floor((drawn_point_index % (density_map.grid[1] * density_map.grid[2])) / density_map.grid[2]) , drawn_point_index % density_map.grid[2] ]
            self.W = iterate(self.W,t,total_iteration_num,drawn_point,epsilon_i,epsilon_f,lambda_i,lambda_f)
              
    def voronoi(self):
         
        # check that grid has been initialized
        if self.status < 1 :
            raise Exception("You can't tessellate a non-existance grid. InitGrid first!") 
        elif self.status >= 2 :
            raise Exception("You have already calculated the Voronoi properties!")
          
        # make enclosing shell of cells around grid to prevent edge distortion in the Voronoi tessellation
        enclose_cell_num = map(int,30 * np.array(self.grid_dim_xyz) * 1.0 / max(self.grid_dim_xyz))  	# number of enclosing cells along each dimension
        shell_dist = 0.2 * max(self.grid_dim_xyz)         				# distance of enclosing cells from system edges
        total_enclose_cell_num = 2 * enclose_cell_num[0] * enclose_cell_num[1] + 2 * enclose_cell_num[0] * (enclose_cell_num[2] - 2) + 2 * (enclose_cell_num[1]-2) * (enclose_cell_num[2]-2) # total number of enclosing cells required
        enclose_cell_dist = ( 2 * shell_dist + np.array(self.grid_dim_xyz) ) / (np.array(enclose_cell_num) - 1)	# separation between enclosing cells
        enclose_cells = np.zeros([total_enclose_cell_num,3])	# initialize array containing enclosing cell node positions 
        p = -1
         
        for i in range(enclose_cell_num[0]):
            for j in range(enclose_cell_num[1]):
                p = p+1
                enclose_cells[p,:] = [i*enclose_cell_dist[0] , j*enclose_cell_dist[1] , 0]
                p = p+1
                enclose_cells[p,:] = [i*enclose_cell_dist[0] , j*enclose_cell_dist[1] , enclose_cell_num[2]-1]
         
        for i in range(enclose_cell_num[0]):
            for k in range(1,enclose_cell_num[2]-1):
                p = p+1
                enclose_cells[p,:] = [i*enclose_cell_dist[0] , 0 , k*enclose_cell_dist[2]]
                p = p+1
                enclose_cells[p,:] = [i*enclose_cell_dist[0] , enclose_cell_num[1]-1 , k*enclose_cell_dist[2]]
         
        for j in range(1,enclose_cell_num[1]-1):
            for k in range(1,enclose_cell_num[2]-1):
                p = p+1
                enclose_cells[p,:] = [ 0 , j*enclose_cell_dist[1] , k*enclose_cell_dist[2] ]
                p = p+1
                enclose_cells[p,:] = [ enclose_cell_num[0]-1 , j*enclose_cell_dist[1], k*enclose_cell_dist[2] ]
         
        self.xW = self.W 					# clone grid node position matrix (we will append the enclosing cells to the clone xW and save the original matrix W)
        self.xW = np.append(self.xW,enclose_cells,axis=0)	# append enclosing cells to clone 
        vorW = spatial.Voronoi(self.xW,qhull_options="QJ")	# run QHull algorithm to obtain Voronoi tessellation
         
        ## Construct neighbor lists so that neighbors[i] returns list of neighbors of node i
        self.neighbors = [ [] for i in range(len(self.xW))]     # initialize neighbors list as list of empty lists with len(self.xW) components
        self.facearea = [ [] for i in range(len(self.xW))]	# initialize face area list such that facearea[i][j] is the interface area between points i and j
        self.volume = np.zeros(len(self.xW))			# initialize volume list such that volume[i] is the volume of cell i
        self.facedist = [ [] for i in range(len(self.xW))]	# initialize face distance list such that facedist[i][j] is the distance of points i and j from their interface
         
        ## Construct neighbor list, face areas, face distances. Iterate through neighbor pairs list
        ## generated by scipy.spatial.Voronoi(); for each pair, calculate the interfacial area and 
        ## distance-from-interface between the pair, populate the neighbors, face areas, face distances
        ## lists indexed by cell number. e.g. Cell i has neighbor list self.neighbors[i] and the 
        ## interfacial area between i and its jth neighbor self.neighbors[i][j] is self.facearea[i][j]
        for i in range(len(vorW.ridge_points)):				
            self.neighbors[vorW.ridge_points[i,0]].append(vorW.ridge_points[i,1])
            self.neighbors[vorW.ridge_points[i,1]].append(vorW.ridge_points[i,0])
            fdist = np.linalg.norm( vorW.points[vorW.ridge_points[i,0]] - vorW.points[vorW.ridge_points[i,1]] ) / 2
            self.facedist[vorW.ridge_points[i,0]].append(fdist)
            self.facedist[vorW.ridge_points[i,1]].append(fdist)
            sorted_vertices = order_arrange_3D(vorW.vertices[vorW.ridge_vertices[i]]) # use custom function order_arrange_3D() to get vertices in rotational order about the cell center
            farea = 0
            face_center = sum(sorted_vertices) / len(sorted_vertices)
            for j in range(len(sorted_vertices)):				      # interfacial area is the sum of areas of triangles each consisting of two adjacent polygon vertices and the polygon center 
                V0 = sorted_vertices[j] - face_center
                V1 = sorted_vertices[(j+1) % len(sorted_vertices)] - face_center
                farea = farea + np.linalg.norm(np.cross(V0,V1)) / 2
            self.facearea[vorW.ridge_points[i,0]].append(farea)
            self.facearea[vorW.ridge_points[i,1]].append(farea)
         
        ## calculate volume as the sum of conic volumes from each face to the cell center.
        for i in range(len(self.xW)):
            self.volume[i] = sum(np.array(self.facearea[i])*np.array(self.facedist[i]))/3
        self.status = 2
      
    ## Assign potentials to Voronoi cells based on potential map, using cubic spline interpolation 
    def assign_potential(self,pot_array,pot_res):
         
        # check that Voronoi calculations have been done 
        if self.status < 2 :
            raise Exception("There will be big problems if cell potentials are assigned before Voronoi geometry is calculated.") 
         
        ## check if input potential map can fit into grid
        if sum(np.array(pot_array.shape) * pot_res > self.grid_dim_xyz) > 0:
            raise Exception("Potential map dimensions must fit inside Voronoi grid!")
         
        ## parse pot_array into form understandable by scipy.interpolate.griddata
        points = np.zeros([np.prod(pot_array.shape),3])
        values = np.zeros([np.prod(pot_array.shape),])
        p = -1
        for i in range(pot_array.shape[0]):
            for j in range(pot_array.shape[1]):
                for k in range(pot_array.shape[2]):
                    p = p + 1
                    points[p,:] = ([ i,j,k ] - np.array(pot_array.shape)/2) * pot_res + self.grid_center
                    values[p] = pot_array[i,j,k]  
         
        self.U = interp.griddata(points,values,self.xW,method='linear')
        self.status = 3
          
    ## Assign potentials to Voronoi cells based on potential map, using cubic spline interpolation
    def assign_potential_map(self,potential_map):
            
        # check that Voronoi calculations have been done
        if self.status < 2 :
            raise Exception("There will be big problems if cell potentials are assigned before Voronoi geometry is calculated.")
           
        ## check if input potential map can fit into grid
        if sum(np.array(potential_map.data.shape) * potential_map.delta > self.grid_dim_xyz) > 0:
            raise Exception("Potential map dimensions must fit inside Voronoi grid!")
           
        ## parse pot_array into form understandable by scipy.interpolate.griddata
        points = np.zeros([potential_map.numval,3])
        values = np.zeros([potential_map.numval,])
        p = -1
        for i in range(potential_map.grid[0]):
            for j in range(potential_map.grid[1]):
                for k in range(potential_map.grid[2]):
                    p = p + 1
                    points[p,:] = ([ i,j,k ] - np.array(map(float,potential_map.grid))/2) * potential_map.delta + self.grid_center
                    values[p] = potential_map.data[i,j,k]
            
        self.U = interp.griddata(points,values,self.xW,method='linear')
        self.status = 3
          
    ## Mark out cells to exclude from grid, or inaccessible to diffusing particles and then remove them from memory
    def exclude_cells(self,pot_threshold):
        if self.status < 3:
            raise Exception("You have not assigned potential values to the cells. Go back and assign_potential.")
        self.celltype = np.ones([len(self.xW),])
        self.celltype[range(int(self.num_nodes),len(self.xW))] = 0
        self.celltype[self.U > pot_threshold] = 0
        self.celltype[np.isnan(self.U)] = 0 
          
        for i in range(int(self.num_nodes)):
            # Change self.facearea[i] to ndarray type to make use of conditional slicing functionality (absent in lists).
            # Extract slice corresponding to neighbors whose celltypes are 1, so that only faceareas of these neighbors
            # are retained. Then convert the ndarray of retained cells back to a list to preserve original type.
            self.facearea[i] = list(np.array(self.facearea[i])[self.celltype[self.neighbors[i]] == 1])     
            self.facedist[i] = list(np.array(self.facedist[i])[self.celltype[self.neighbors[i]] == 1])
            self.neighbors[i] = np.array(self.neighbors[i])[self.celltype[self.neighbors[i]] == 1]
        self.status = 4
          
    ## calculate probability matrix elements - inputs are diffusion coefficient (constant) and temperature
    def make_rate_mat(self,D,temp):
        if self.status < 4:
            raise Exception("You have not excluded reflective cells. Go back and exclude_cells.")
        self.D = D
        self.temp = temp
        self.rate_mat = [ [] for i in range(int(self.num_nodes))] 			# initialize rate matrix (actually a list of lists of variable lengths)
        self.rate_mat_out = [ [] for i in range(int(self.num_nodes))]                       # initialize rate matrix (actually a list of lists of variable lengths)
        # rate_mat[i][j] is the rate of probability flow from the j-th neighbor, self.neighbors[i][j], into i
        for i in range(int(self.num_nodes)):
            if self.celltype[i] == 1:
                self.rate_mat[i] = D * np.array(self.facearea[i]) / (2 * np.array(self.facedist[i]) * self.volume[self.neighbors[i]])  * np.exp( (- self.U[i] + self.U[self.neighbors[i]]) / 2 )
                self.rate_mat_out[i] = D * np.array(self.facearea[i]) / (2 * np.array(self.facedist[i]) * self.volume[i])  * np.exp( ( self.U[i] - self.U[self.neighbors[i]]) / 2 )
        self.status = 5    
      
    ## calculate transition matrix elements
    def make_trans_mat(self,dt):
        if self.status < 5:
            raise Exception("Are you trying to calculate the transition matrix from thin air? Go back and make_rate_mat.")
        diff_range = 3 * (6 * self.D * dt) ** (0.5)					# calculate the diffusion range within a time step. Particles will diffuse only to cells within this range in one timestep.
        search_cubes = cube_sys(self.W,2*diff_range)					# use the cube_sys function to split the system into several cubes so that search for neighbors can be confined to a small number of cells.
        self.trans_mat = [ [] for i in range(int(self.num_nodes))]				# initialize transition matrix
        self.neighbor_pool = [ [] for i in range(int(self.num_nodes))]			# initialize pool of cells that are within diffusion range
        for i in range(int(self.num_nodes)):
            if self.celltype[i] == 0:
                continue
            global submatrix
            self.neighbor_pool[i] = [i]							# first add the cell to its own neighbor pool
            self.neighbor_pool[i] = np.append(self.neighbor_pool[i],self.neighbors[i])  # then add its immediate neighbors (these may sometimes be outside diffusion range but should be added anyway)
            self.neighbor_pool[i] = np.append(self.neighbor_pool[i],[ j for j in search_cubes.nodes_in_cube[int(search_cubes.cube_of[i])] if np.linalg.norm(self.W[i] - self.W[j])<diff_range and j not in self.neighbor_pool[i]] and self.celltype[j] == 1)	# now add the rest of the cells within range
            submatrix = np.zeros([len(self.neighbor_pool[i]),len(self.neighbor_pool[i])])	# initialize submatrix of rates
            inverse_pool = dict()								# create a dictionary that maps cell numbers to their indices in the neighbor pool
            for j in range(len(self.neighbor_pool[i])):						 
                inverse_pool[self.neighbor_pool[i][j]] = j
            for j in self.neighbor_pool[i]:						# for each cell in neighbor pool, for each neighbor that is also in the pool, assign the appropriate transition rate value from the whole-system rate matrix to the corresponding element in the submatrix 
                j = int(j)
                for k in range(len(self.neighbors[j])):
                    if self.neighbors[j][k] in self.neighbor_pool[i]:
                        submatrix[inverse_pool[j],inverse_pool[self.neighbors[j][k]]] = self.rate_mat[j][k]
            for j in self.neighbor_pool[i]:      
                submatrix[inverse_pool[j],inverse_pool[j]] = - sum( submatrix[:,inverse_pool[j]] )
            P_init = np.zeros([len(self.neighbor_pool[i]),])
            P_init[0] = 1
            def sub_rate_eq(y,t0):
                return np.dot(submatrix,y)
            int_result = integr.odeint(sub_rate_eq,P_init,[0,dt])			# integrate the rate equation represented by the submatrix
            self.trans_mat[i] = int_result[1]   					# assign the integration result to the corresponding transition matrix entries 
                         


