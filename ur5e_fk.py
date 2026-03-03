import numpy as np

# ---- UR5e DH Constants ----
d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

def dh_matrix(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),              d],
        [0,              0,                           0,                          1]
    ])

def forward_kinematics(q):
    theta1, theta2, theta3, theta4, theta5, theta6 = q

    T01 = dh_matrix(0, np.pi/2, d1, theta1)
    T12 = dh_matrix(a2, 0, 0, theta2)
    T23 = dh_matrix(a3, 0, 0, theta3)
    T34 = dh_matrix(0, np.pi/2, d4, theta4)
    T45 = dh_matrix(0, -np.pi/2, d5, theta5)
    T56 = dh_matrix(0, 0, d6, theta6)

    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    return T06

# Example test
#- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
#
q_test = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
T = forward_kinematics(q_test)

print("T06 =")
print(T)
print("\nEnd Effector Position (x,y,z):")
print(T[0:3, 3])

