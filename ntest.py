import random
import numpy as np

current_pose=np.array([1,3,5])
Node = np.zeros(3)


def get_random_node(current_pose):
    rnd = np.array([random.uniform(current_pose[0]-1, current_pose[0]+1),
        random.uniform(current_pose[1]-1, current_pose[1]+1),
        random.uniform(current_pose[2]-1, current_pose[2]+1)])
    print(np.random.rand(3))
    return rnd

if __name__ == '__main__':
    get_random_node(current_pose)