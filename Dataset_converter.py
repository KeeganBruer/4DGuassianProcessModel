import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np

def convert_dataset(origin, w, h, view_distance=10, world_q=[]):
    points = []
    for x in range(w):
        for y in range(h):
            shift_x = x - (w/2) #Shift so 0,0 is in the center
            shift_y = y - (h/2)
            pixel_q = get_quaternion(shift_x, shift_y, w, h, 1.0472) #get quaternion of pixel
            #quaternion in terms of the world origin
            q = quaternion_multiply(pixel_q, world_q)
            #q = pixel_q
            forward = [
                2 * (q[1]*q[3] + q[0]*q[2]),  # X
                2 * (q[2]*q[3] + q[0]*q[1]),  # Y
                1- 2 * (q[1]*q[1] + q[2]*q[2])# Z
            ]
            new_point = [
                origin[0] + view_distance * forward[0],
                origin[1] + view_distance * forward[1],
                origin[2] + view_distance * forward[2]
            ]
            points.append(new_point)
    return points
def get_quaternion(x, y, w, h, fov):
    x_angle = (x/float(w))*fov #Get angle if x is -w/2 then x/w is -1/2. then the angle is -1/2 * fov (60deg) = -30deg.
    y_angle = (y/float(h))*fov
    #print("{0:.2f} {1:.2f} \n".format(x_angle, y_angle), end="")
    xq = get_quaternion_about([1, 0, 0], x_angle)
    yq = get_quaternion_about([0, 1, 0], y_angle)
    zq = get_quaternion_about([0, 0, 1], 0)
    q = quaternion_multiply(quaternion_multiply(xq, yq), zq)
    #print(q)
    return q
def get_quaternion_about(axis, angle):
    factor = math.sin(angle/2)
    x = axis[0] * factor
    y = axis[1] * factor
    z = axis[2] * factor
    w = math.cos(angle/2)
    return [w, x, y, z]


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


if __name__ == "__main__":
    #data = np.load("./training_data/sample_set_0.npz")
    #print(data.files)
    #print(data["sample_set_x"])
    #print(data["sample_set_y"])
    origin = [0, 0, 0]
    world_q = [1, 0, 0, 0]
    view_distance = 10
    w = 100
    h = 100
    points = convert_dataset(origin, w, h, view_distance, world_q)
    
    fig = go.Figure()
    #fig.update_xaxes(range=[140, 190])
    #fig.update_traces(mode='markers')
    fig.add_trace(go.Scatter3d(
        x=[point[0] for point in points],
        y=[point[1] for point in points],
        z=[point[2] for point in points],
        mode='markers',
    ))

    fig.show()

