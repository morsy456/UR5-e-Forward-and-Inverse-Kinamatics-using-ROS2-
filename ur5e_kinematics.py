import numpy as np

# =============================
# UR5e DH PARAMETERS (Standard DH)
# =============================

d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

def dh(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),              d],
        [0,              0,                           0,                          1]
    ])

def forward_kinematics(q):
    T01 = dh(0, np.pi/2, d1, q[0])
    T12 = dh(a2, 0, 0, q[1])
    T23 = dh(a3, 0, 0, q[2])
    T34 = dh(0, np.pi/2, d4, q[3])
    T45 = dh(0, -np.pi/2, d5, q[4])
    T56 = dh(0, 0, d6, q[5])
    return T01 @ T12 @ T23 @ T34 @ T45 @ T56

# =============================
# JACOBIAN (Numerical)
# =============================

def compute_jacobian(q):
    J = np.zeros((6,6))
    T = np.eye(4)
    z = []
    o = []

    transforms = [
        dh(0, np.pi/2, d1, q[0]),
        dh(a2, 0, 0, q[1]),
        dh(a3, 0, 0, q[2]),
        dh(0, np.pi/2, d4, q[3]),
        dh(0, -np.pi/2, d5, q[4]),
        dh(0, 0, d6, q[5])
    ]

    o.append(T[:3,3])
    z.append(T[:3,2])

    for Ti in transforms:
        T = T @ Ti
        o.append(T[:3,3])
        z.append(T[:3,2])

    o_e = o[-1]

    for i in range(6):
        Jp = np.cross(z[i], o_e - o[i])
        Jo = z[i]
        J[:3,i] = Jp
        J[3:,i] = Jo

    return J

# =============================
# INVERSE KINEMATICS (Numerical)
# =============================

def inverse_kinematics(target_pos, q_init, iterations=1000, alpha=0.5):
    q = np.array(q_init, dtype=float)

    for _ in range(iterations):
        T = forward_kinematics(q)
        pos = T[:3,3]
        error = target_pos - pos

        if np.linalg.norm(error) < 1e-4:
            return q

        J = compute_jacobian(q)
        Jp = np.linalg.pinv(J[:3,:])
        dq = alpha * Jp @ error
        q += dq

    return q

# =============================
# MAIN TEST
# =============================

if __name__ == "__main__":

    print("\n=== FORWARD KINEMATICS TEST ===")
    q_test = [0.0, -1.0, 1.0, 0.0, 0.5, 0.0]
    T = forward_kinematics(q_test)

    print("T06 =\n", T)
    print("Position =", T[:3,3])

    print("\n=== INVERSE KINEMATICS TEST ===")
    target = T[:3,3]
    q_guess = [0,0,0,0,0,0]
    q_sol = inverse_kinematics(target, q_guess)

    print("Recovered q =", q_sol)


