import math
##################
# Ref. : https://github.com/AtsushiSakai/PythonRobotics
##################

class AStarPlanner:
    def __init__(self, ox, oy, oz, resolution, rr):

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y, self.min_z = 0, 0, 0
        self.max_x, self.max_y, self.max_z = 0, 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width, self.z_width = 0, 0, 0

        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy, oz)

    class Node:
        def __init__(self, x, y, z, cost, parent_index):
            self.x = x
            self.y = y
            self.z = z
            self.cost = cost
            self.parent_index = parent_index
        
        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.z) + "," + str(self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, sz, gx, gy, gz):
        start_node = self.Node(self.calc_xyz_index(sx, self.min_x), self.calc_xyz_index(sy, self.min_y), self.calc_xyz_index(sz, self.min_z), 0.0, -1)
        goal_node = self.Node(self.calc_xyz_index(gx, self.min_x), self.calc_xyz_index(gy, self.min_y), self.calc_xyz_index(gz, self.min_z), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("start_node is empty...")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))

            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y and current.z == goal_node.z:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break
                
            del open_set[c_id]

            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = self.Node(\
                    current.x + self.motion[i][0],\
                    current.y + self.motion[i][1],\
                    current.z + self.motion[i][2],\
                    current.cost + self.motion[i][3],
                    c_id\
                )
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue
                
                if n_id in closed_set:
                    continue
                
                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node
        
        ##### Road Map --> closed_set #####
        rx, ry, rz = self.calc_final_path(goal_node, closed_set)
        return rx, ry, rz


            

    
    ########################################## INIT #################################################
    def calc_obstacle_map(self, ox, oy, oz):
        #min part
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.min_z = round(min(oz))
        #max part
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        self.max_z = round(max(oz))
        #width
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.z_width = round((self.max_z - self.min_z) / self.resolution)

        #Obstacle map
        self.obstacle_map = [[[False for _ in range(self.z_width)] for _ in range(self.y_width)] for _ in range(self.x_width)]
        print("1-1")
        for ix in range(self.x_width):
            x = self.cal_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.cal_grid_position(iy, self.min_y)
                for iz in range(self.z_width):
                    z = self.cal_grid_position(iz, self.min_z)
                    for iox, ioy, ioz in zip(ox, oy, oz):
                        d = math.sqrt((iox-x)**2 + (ioy-y)**2 + (ioz-z)**2)
                        print(d)
                        if d<=self.rr:
                            self.obstacle_map[ix][iy][iz] = True
                            break
        print("1-2")
    def get_motion_model(self):
        motion = [
            # one axis move
            [1,0,0,1],
            [-1,0,0,1],

            [0,1,0,1],
            [0,-1,0,1],

            [0,0,1,1],
            [0,0,-1,1],

            # two axises move
            [1,1,0,math.sqrt(2)],
            [-1,1,0,math.sqrt(2)],
            [1,-1,0,math.sqrt(2)],
            [-1,-1,0,math.sqrt(2)],

            [1,0,1,math.sqrt(2)],
            [-1,0,1,math.sqrt(2)],
            [1,0,-1,math.sqrt(2)],
            [-1,0,-1,math.sqrt(2)],

            [0,1,1,math.sqrt(2)],
            [0,-1,1,math.sqrt(2)],
            [0,1,-1,math.sqrt(2)],
            [0,-1,-1,math.sqrt(2)],

            # three axises move
            [1,1,1,math.sqrt(3)],

            [-1,1,1,math.sqrt(3)],
            [1,-1,1,math.sqrt(3)],
            [1,1,-1,math.sqrt(3)],

            [-1,-1,1,math.sqrt(3)],
            [-1,1,-1,math.sqrt(3)],
            [1,-1,-1,math.sqrt(3)],

            [-1,-1,-1,math.sqrt(3)]
        ]
        return motion

    ########################################## UTILS #################################################
    def cal_grid_position(self, idx, min_val):
        return idx*self.resolution + min_val
    
    def calc_xyz_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        idx = (node.x - self.min_x)*self.y_width*self.z_width\
            + (node.y - self.min_y)*self.x_width*self.z_width\
            + (node.z - self.min_x)*self.x_width*self.y_width
        return idx
    
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0
        d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2 + (n1.z - n2.z)**2)
        return d

    def verify_node(self, node):
        px = self.cal_grid_position(node.x, self.min_x)
        py = self.cal_grid_position(node.y, self.min_y)
        pz = self.cal_grid_position(node.z, self.min_z)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif pz < self.min_z:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False
        elif pz >= self.max_z:
            return False
        
        if self.obstacle_map[node.x][node.y][node.z]:
            return False
        
        return True
    
    def calc_final_path(self, goal_node, closed_set):
        rx, ry, rz = [ self.cal_grid_position(goal_node.x, self.min_x) ],\
                     [ self.cal_grid_position(goal_node.y, self.min_y) ],\
                     [ self.cal_grid_position(goal_node.z, self.min_z) ]
        parent_index = goal_node.parent_index
        while parent_index != -1 :
            n = closed_set[parent_index]
            rx.append(self.cal_grid_position(n.x, self.min_x))
            ry.append(self.cal_grid_position(n.y, self.min_y))
            rz.append(self.cal_grid_position(n.z, self.min_z))
            parent_index = n.parent_index
            print(rx, ry, rz)
        return rx, ry, rz


def main():
    sx=10.0
    sy=10.0
    sz=10.0

    gx=50.0
    gy=50.0
    gz=50.0

    ox, oy, oz = [], [], []
    for i_1 in range(-10, 60):
        for i_2 in range(-10, 60):
            ox.append(i_1)
            oy.append(i_2)
            oz.append(-10.0)

    for i_1 in range(-10, 60):
        for i_2 in range(-10, 60):
            oy.append(i_1)
            oz.append(i_2)
            ox.append(-10.0)

    for i_1 in range(-10, 60):
        for i_2 in range(-10, 60):
            oz.append(i_1)
            ox.append(i_2)
            oy.append(-10.0)

    for i_1 in range(-10, 60):
        for i_2 in range(-10, 60):
            ox.append(i_1)
            oy.append(i_2)
            oz.append(60.0)
    
    for i_1 in range(-10, 60):
        for i_2 in range(-10, 60):
            oy.append(i_1)
            oz.append(i_2)
            ox.append(60.0)

    for i_1 in range(-10, 60):
        for i_2 in range(-10, 60):
            oz.append(i_1)
            ox.append(i_2)
            oy.append(60.0)
    
    print("1")
    a_star = AStarPlanner(ox, oy, oz, resolution=2.0, rr=1.0)
    print("2")
    rx, ry, rz = a_star.planning(sx, sy, sz, gx, gy, gz)


if __name__ == '__main__':
    main()