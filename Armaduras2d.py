import numpy as np

def K_global(nodos):
    GDL = 2 * nodos
    K = np.zeros((GDL, GDL))
    return K

def elemento_barra_2D(E, A, L, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c*c, c*s],
                  [c*s, s*s]])
    k = ((E * A) / L) * np.block([[R, -R],
                                  [-R, R]])
    return k

def ensamblaje(K_global, k_e, nodo_i, nodo_j):
     nodo_i -= 1
     nodo_j -= 1
     K_global[2*nodo_i:2*nodo_i+2, 2*nodo_i:2*nodo_i+2] += k_e[0:2, 0:2]
     K_global[2*nodo_i:2*nodo_i+2, 2*nodo_j:2*nodo_j+2] += k_e[0:2, 2:4]
     K_global[2*nodo_j:2*nodo_j+2, 2*nodo_i:2*nodo_i+2] += k_e[2:4, 0:2]
     K_global[2*nodo_j:2*nodo_j+2, 2*nodo_j:2*nodo_j+2] += k_e[2:4, 2:4]
     return K_global


def vector_F(nodos):
    GDL = 2 * nodos
    F = np.zeros(GDL)
    return F    

def aplicar_cargas(F, P, nodo, Fx, Fy): # si x=1 se aplica en esa dirección, si x=0 no se aplica. P es la carga
    nodo -= 1
    if Fx == 1:
        F[2*nodo] = P
    elif Fy == 1:
        F[2*nodo + 1] = P
    return F

def restricciones(K_global, F, cond): #si x o y = 1, no está restringido y si x o y = 0, está restringido
    eliminar = []
    for nodo, x, y in cond:
        nodo -= 1
        dof_x = 2 * nodo
        dof_y = 2 * nodo + 1
        if x == 0:
            eliminar.append(dof_x)
        if y == 0:
            eliminar.append(dof_y)
        eliminar = sorted(eliminar, reverse=True)

    Kr = np.delete(K_global, eliminar, axis=0)
    Kr = np.delete(Kr, eliminar, axis=1)
    Fr = np.delete(F, eliminar, axis=0) 

    return Kr, Fr

def vector_q(u, cond, nodos):
    q = []
    i = 0
    for nodo, x, y in cond:
        nodo -= 1
        if x == 1:
            q.append(u[i])
        elif x ==0:
            q.append(0)
        if y == 1:
            q.append(u[i])  
        elif y ==0:
            q.append(0)
        i+=1
    return np.array(q)

def esfuerzos_barra(E, theta, q, nodo_i, nodo_j, L):
    nodo_i -= 1
    nodo_j -= 1
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, s, 0, 0], [0, 0, c, s]])
    u_e = np.array([q[2*nodo_i], q[2*nodo_i + 1], q[2*nodo_j], q[2*nodo_j + 1]])
    B = (1/L) * np.array([-1, 1])
    sigma = E*B @ R @ u_e
    return sigma

# Párametros del problema
E = 200e9  # Módulo de Young en Pa 
A, A_3 = 62.5e-4, 125e-4  # Áreas transversales en m^2
L_1, L_2, L_3 = 5, 4, 3 # Longitudes de las barras en m
theta_1, theta_2, theta_3 = np.arctan(3/4), 0, np.pi/2  # Ángulos en radianes

# Mátrices de rígidez de los elemento
k_e1, k_e2, k_e3 = elemento_barra_2D(E, A, L_1, theta_1), elemento_barra_2D(E, A, L_2, theta_2), elemento_barra_2D(E, A_3, L_3, theta_3) # (función, E, A, L, theta)

# Ensamblaje de la matriz de rigidez global
K = ensamblaje(K_global(3), k_e1, 1, 3) # (K global, k_e, nodo_i, nodo_j)
K = ensamblaje(K, k_e2, 1, 2) # (K global, k_e, nodo_i, nodo_j)
K = ensamblaje(K, k_e3, 2, 3) # (K global, k_e, nodo_i, nodo_j)

# Vector de fuerzas
P = -100e3  # Carga aplicada en N
F = aplicar_cargas(vector_F(3), P/2, 2, 0, 1) # (función, valor carga, nodo, Fx, Fy)

# Aplicación de restricciones
cond = [(1, 1, 0),  # (nodo, x, y)
        (2, 0, 1),  # (nodo, x, y)
        (3, 0, 1)]  # (nodo, x, y)

Kr, Fr = restricciones(K, F, cond) # (rigidez global, vector fuerza, condiciones de frontera)

# Cálculo de desplazamientos
u = np.linalg.solve(Kr, Fr) # u = Kr\Fr
print("Desplazamientos nodales (m):")
print(u)

q = vector_q(u, cond, 3)  # (desplazamientos resueltos, condiciones de frontera, número de nodos)
print("Vector de desplazamientos completo q (m):")
print(q)

# Cálculo de reacciones
R = K @ q - F 
print("Reacciones en los apoyos R (N):")
print(R)

# Calculo de esfuerzos en las barras    
esf1 = esfuerzos_barra(E, theta_1, q, 1, 3, L_1) # (E, theta, q, nodo_i, nodo_j, L)
esf2 = esfuerzos_barra(E, theta_2, q, 1, 2, L_2) # (E, theta, q, nodo_i, nodo_j, L)
esf3 = esfuerzos_barra(E, theta_3, q, 2, 3, L_3) # (E, theta, q, nodo_i, nodo_j, L)
print("Esfuerzos en las barras (N):")   
print(f"Esfuerzo en barra 1 (nodos 1-3): {esf1:.2f} Pa")
print(f"Esfuerzo en barra 2 (nodos 1-2): {esf2:.2f} Pa")
print(f"Esfuerzo en barra 3 (nodos 2-3): {esf3:.2f} Pa")