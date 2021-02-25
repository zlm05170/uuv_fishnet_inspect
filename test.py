"""
Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)
author: AtsushiSakai(@Atsushi_twi)
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.parent = None

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0] #I think that these should not change
        # The reason it that,the same parameters can be used in z-direction
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)
            #print(self.node_list[-1].y)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y,
                                      self.node_list[-1].z) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y, from_node.z)
        d, theta, alpha = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.path_z = [new_node.z]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.z += self.path_resolution * math.sin(alpha)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            new_node.path_z.append(new_node.z)

        d, _, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.path_z.append(to_node.z)
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.z = to_node.z

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y, self.end.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])

        return path

    def calc_dist_to_goal(self, x, y, z):
        dx = x - self.end.x
        dy = y - self.end.y
        dz = z - self.end.z
        s = math.hypot(dx, dy)
        r = math.hypot(s, dz)
        return r

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.z)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.gca(projection='3d').plot(rnd.x, rnd.y, rnd.z, "^k")
        for node in self.node_list:
            if node.parent:
                plt.gca(projection='3d').plot(node.path_x, node.path_y, node.path_z, "-g")

        for (ox, oy, oz, size) in self.obstacle_list:
            self.plot_circle(ox, oy, oz, size)

        plt.gca(projection='3d').plot(self.start.x, self.start.y, self.start.z, "xr")
        plt.gca(projection='3d').plot(self.end.x, self.end.y, self.end.z, "xr")
        plt.axis("auto")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)


    """
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        zl =
        plt.plot(xl, yl, color)
"""
    @staticmethod
    def plot_circle(x, y, z, size):

        u = np.linspace(0, 2 * np.pi, 72)
        v = u
        xl = x + size * np.outer(np.cos(u), np.sin(v))
        yl = y + size * np.outer(np.sin(u), np.sin(v))
        zl = z + size * np.outer(np.ones(np.size(u)), np.cos(v))

        plt.gca(projection='3d').plot_wireframe(xl, yl, zl, color = 'k')


    @staticmethod
    # I think that these are the metrics
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 + (node.z - rnd_node.z)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False

        for (ox, oy, oz, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            dz_list = [oz - z for z in node.path_z]
            d_list = [dx * dx + dy * dy + dz * dz for (dx, dy, dz) in zip(dx_list, dy_list, dz_list)]

            if min(d_list) <= size**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = math.hypot(dx, dy)
        d = math.hypot(d, dz)
        theta = math.atan2(dy, dx)
        alpha = math.atan2(dz, d)
        return d, theta, alpha


def main(gx=6.0, gy=10.0, gz = 0.0):
    n = 2
    gx, gy, gz = [], [], []
    sx, sy, sz = [], [], []
    """
    for i in range(n):
        gx1, gy1, gz1 = [6.0, 10.0], [10.0, 12.0], [0.0, 5.0]
        gx.append(gx1[i]), gy.append(gy1[i]), gz.append(gz1[i])
        sx1, sy1, sz1 = [0.0, gx[0]], [0.0, gy[0]], [0.0, gz[0]]
        sx.append(sx1[i]), sy.append(sy1[i]), sz.append(sz1[i])
    
    gx, gy, gz = gx[0], gy[0], gz[0]
    sx, sy, sz = sx[0], sy[0], sz[0]
    """
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 5, 2), (4, 10, 5, 1)]  # [x, y, z, radius]
    # Set Initial parameters

    rrt = []
    path = []
    for i in range(n):
        gx1, gy1, gz1 = [6.0, 10.0], [10.0, 12.0], [0.0, 5.0]
        gx.append(gx1[i]), gy.append(gy1[i]), gz.append(gz1[i])
        sx1, sy1, sz1 = [0.0, gx[0]], [0.0, gy[0]], [0.0, gz[0]]
        sx.append(sx1[i]), sy.append(sy1[i]), sz.append(sz1[i])
        rrt1 = RRT(
            start=[sx[i], sy[i], sz[i]],
            goal=[gx[i], gy[i], gz[i]],
            rand_area=[-2, 15],
            obstacle_list=obstacleList)
        rrt.append(rrt1)
        path1= rrt[i].planning(animation=show_animation)
        path.append(path1)
        if path1 is None:
            print("Cannot find path")
        else:
            print("found path!!")

            # Draw final path
            if show_animation:
                rrt1.draw_graph()
                plt.gca(projection='3d').plot([x for (x, y, z) in path[i]], [y for (x, y, z) in path[i]], [z for (x, y, z) in path[i]], '-r')
                plt.grid(True)
                plt.pause(0.01)  # Need for Mac
                plt.show()




if __name__ == '__main__':
    main()