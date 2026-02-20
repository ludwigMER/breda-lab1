import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ======================================================================================
# ПОДКЛЮЧЕНИЕ АВТОМАТИЧЕСКОГО ДИФФЕРЕНЦИРОВАНИЯ (JAX)
# ======================================================================================

try:
    import jax
    import jax.numpy as jnp
    from jax import config
    # Включаем высокую точность (float64), чтобы сравнивать с numpy один-в-один
    config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("ВНИМАНИЕ: Библиотека JAX не найдена. Работаем в ручном режиме.")
    print("Для установки: pip install jax jaxlib")

# ======================================================================================
# КОНСТАНТЫ
# ======================================================================================

N = 30      # Длина выборки
n = 2       # Размерность x
m = 1       # Размерность y
r = 1       # Размерность u
p = 1       # Размерность w

TRUE_THETA = np.array([-4.6, 5]) 
THETA1_BOUNDS = np.array([-10.0, -1.5]) # Границы для первого параметра
THETA2_BOUNDS = np.array([1.0, 8.0])    # Границы для второго параметра
# Cписок границ для minimize
THETA_BOUNDS = [tuple(THETA1_BOUNDS), tuple(THETA2_BOUNDS)]

U_BOUNDS = np.array([0.0, 7.0])
USE_JAX_OPTIMIZER = True 

# --- ГЛОБАЛЬНЫЕ КОНСТАНТНЫЕ МАТРИЦЫ ---
CONST_GAMMA = np.array([[1.0], [0.0]])
CONST_H     = np.array([[1.0, 0.0]])
CONST_Q     = np.eye(p) * 0.3
CONST_R     = np.eye(m) * 0.4
CONST_X0    = np.zeros(n)
CONST_P0    = np.eye(n) * 0.2

# --- УНИВЕРСАЛЬНЫЙ СТРОИТЕЛЬ МАТРИЦ ---
def build_parametric_matrices(theta, xp):
    """
    xp - это библиотека (numpy или jax.numpy).
    """
    th1, th2 = theta[0], theta[1]
    
    F = xp.array([
        [0.1 * th1, 1.0], 
        [-0.9, 0.0]
    ])
    
    Psi = xp.array([
        [th2], 
        [0.0]
    ])
    
    return F, Psi

# ======================================================================================
# ПРОВЕРКА СВОЙСТВ СИСТЕМЫ
# ======================================================================================

def check_system_properties(theta):
    """
    Проверяет устойчивость, управляемость и наблюдаемость для заданных параметров.
    """
    print(f"\n--- Проверка свойств системы для theta={theta} ---")
    F, Psi, _, H, _, _, _, _ = get_system_matrices(theta)
    
    # 1. Устойчивость (Собственные числа F по модулю < 1)
    eig_vals = np.linalg.eigvals(F)
    max_eig = np.max(np.abs(eig_vals))
    is_stable = max_eig < 1.0 + 1e-6
    print(f"Собственные числа F: {eig_vals}")
    print(f"Макс. модуль: {max_eig:.4f} -> {'УСТОЙЧИВА' if is_stable else 'НЕУСТОЙЧИВА'}")
    
    # 2. Управляемость: rank([Psi, F*Psi, ...]) = n
    Controllability = Psi
    temp_Psi = Psi
    for _ in range(n - 1):
        temp_Psi = F @ temp_Psi
        Controllability = np.hstack((Controllability, temp_Psi))
        
    rank_C = np.linalg.matrix_rank(Controllability)
    is_controllable = rank_C == n
    print(f"Ранг матрицы управляемости: {rank_C}/{n} -> {'УПРАВЛЯЕМА' if is_controllable else 'НЕУПРАВЛЯЕМА'}")

    # 3. Наблюдаемость: rank([H, H*F, ...]^T) = n
    Observability = H
    temp_H = H
    for _ in range(n - 1):
        temp_H = temp_H @ F
        Observability = np.vstack((Observability, temp_H))
        
    rank_O = np.linalg.matrix_rank(Observability)
    is_observable = rank_O == n
    print(f"Ранг матрицы наблюдаемости: {rank_O}/{n} -> {'НАБЛЮДАЕМА' if is_observable else 'НЕНАБЛЮДАЕМА'}")
    print("-" * 50 + "\n")

    if not (is_controllable and is_observable):
        print("ПРЕДУПРЕЖДЕНИЕ: Система вырождена. Идентификация может быть неточной.")

    if not (is_stable):
        print("ПРЕДУПРЕЖДЕНИЕ: Система неустойчива. Идентификация может дать неточный результат.")

# ======================================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ДАННЫХ
# ======================================================================================

def get_system_matrices(theta):
    F, Psi = build_parametric_matrices(theta, np)
    return F, Psi, CONST_GAMMA, CONST_H, CONST_Q, CONST_R, CONST_X0, CONST_P0

def get_matrix_derivatives(theta):
    s = len(theta)
    dF = [np.zeros((n, n)) for _ in range(s)]
    dPsi = [np.zeros((n, r)) for _ in range(s)]
    dGamma = [np.zeros((n, p)) for _ in range(s)]
    dH = [np.zeros((m, n)) for _ in range(s)]
    dQ = [np.zeros((p, p)) for _ in range(s)]
    dR = [np.zeros((m, m)) for _ in range(s)]
    dx0 = [np.zeros(n) for _ in range(s)]
    dP0 = [np.zeros((n, n)) for _ in range(s)]
    
    # Производные по theta[0] (th1 в F)
    dF[0][0, 0] = 1.0
    
    # Производные по theta[1] (th2 в Psi)
    dPsi[1][0, 0] = 1.0
    
    return dF, dPsi, dGamma, dH, dQ, dR, dx0, dP0

def get_input_signal(k):
    return np.array([3.0])

def generate_data(theta_true, N_samples):
    F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices(theta_true)
    x = np.random.multivariate_normal(x0, P0)
    Y_obs = []; U_inputs = [] 
    
    for k in range(N_samples):
        u = get_input_signal(k)
        U_inputs.append(u)
        w = np.random.multivariate_normal(np.zeros(p), Q)
        v = np.random.multivariate_normal(np.zeros(m), R)
        y = H @ x + v
        Y_obs.append(y)
        x = F @ x + Psi @ u + Gamma @ w
        
    return np.array(Y_obs), np.array(U_inputs)

# ======================================================================================
# ФУНКЦИИ ДЛЯ JAX (АВТО-ДИФФЕРЕНЦИРОВАНИЕ)
# ======================================================================================

if JAX_AVAILABLE:
    def get_system_matrices_jax(theta):
        """Версия для JAX."""
        F, Psi = build_parametric_matrices(theta, jnp)
        
        Gamma = jnp.array(CONST_GAMMA)
        H     = jnp.array(CONST_H)
        Q     = jnp.array(CONST_Q)
        R     = jnp.array(CONST_R)
        x0    = jnp.array(CONST_X0)
        P0    = jnp.array(CONST_P0)
        return F, Psi, Gamma, H, Q, R, x0, P0

    def log_likelihood_and_gradient_jax(theta : np.ndarray, Y_obs : jax.Array, U_inputs : jax.Array):
        """
        Чистая функция расчета критерия J (без ручного градиента).
        JAX автоматически строит граф вычислений для этой функции.
        """
        F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices_jax(theta)
        
        x_pred = x0; P_pred = P0
        x_filt = x0; P_filt = P0
        
        J = 0.0
        J += 0.5 * N * m * jnp.log(2 * jnp.pi)
        
        for k in range(N):
            u = U_inputs[k]
            y_obs = Y_obs[k]
            
            if k > 0:
                 x_pred = F @ x_filt + Psi @ u
                 P_pred = F @ P_filt @ F.T + Gamma @ Q @ Gamma.T

            epsilon = y_obs - H @ x_pred
            B = H @ P_pred @ H.T + R
            
            _, log_det_B = jnp.linalg.slogdet(B)
            term_quad = epsilon.T @ jnp.linalg.solve(B, epsilon)
            
            J += 0.5 * (log_det_B + term_quad)
            
            K = P_pred @ H.T @ jnp.linalg.inv(B)
            x_filt = x_pred + K @ epsilon
            P_filt = (jnp.eye(n) - K @ H) @ P_pred
            
        return J

    def create_jax_optimization_wrapper(Y_obs : np.ndarray, U_inputs : np.ndarray):
        """
        Создает функцию-обертку для scipy.minimize.
        1. Компилирует (JIT) функцию value_and_grad.
        2. Замораживает данные (Y и U) внутри замыкания.
        3. Конвертирует типы данных между Numpy (scipy) и JAX.
        """
        Y_jax = jnp.array(Y_obs)
        U_jax = jnp.array(U_inputs)

        jax_val_and_grad = jax.jit(jax.value_and_grad(log_likelihood_and_gradient_jax))
        
        def objective_function(theta : np.ndarray):
            # Вычисляем через JAX
            val, grad = jax_val_and_grad(theta, Y_jax, U_jax)
            
            # Конвертируем обратно
            return float(val), np.array(grad)
            
        return objective_function

# ======================================================================================
# ОСНОВНЫЕ ФУНКЦИИ (РУЧНОЙ МЕТОД)
# ======================================================================================

def log_likelihood_and_gradient(theta, Y_obs : np.ndarray, U_inputs : np.ndarray):
    """
    РУЧНОЙ РАСЧЕТ: Критерий J и аналитический градиент dJ/dtheta.
    """
    try:
        F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices(theta)
        dF, dPsi, dGamma, dH, dQ, dR, dx0, dP0 = get_matrix_derivatives(theta)
    except:
        return 1e10, np.zeros(len(theta))

    s = len(theta)
    x_pred = x0.copy(); P_pred = P0.copy()
    x_filt = x0.copy(); P_filt = P0.copy()
    
    dx_pred = [dx0[i].copy() for i in range(s)]
    dP_pred = [dP0[i].copy() for i in range(s)]
    dx_filt = [dx0[i].copy() for i in range(s)]
    
    J = 0.0
    grad_J = np.zeros(s)
    J += 0.5 * N * m * np.log(2 * np.pi)
    
    for k in range(N):
        u = U_inputs[k]
        y_obs = Y_obs[k]
        
        if k > 0:
            x_pred = F @ x_filt + Psi @ u
            P_pred = F @ P_filt @ F.T + Gamma @ Q @ Gamma.T
            for a in range(s):
                dx_pred[a] = dF[a] @ x_filt + F @ dx_filt[a] + dPsi[a] @ u
                dP_pred[a] = (dF[a] @ P_filt @ F.T + F @ dP_filt[a] @ F.T + 
                              F @ P_filt @ dF[a].T + dGamma[a] @ Q @ Gamma.T + 
                              Gamma @ dQ[a] @ Gamma.T + Gamma @ Q @ dGamma[a].T)

        epsilon = y_obs - H @ x_pred
        B = H @ P_pred @ H.T + R
        
        dB = []
        dEps = []
        for a in range(s):
            dEps.append(-dH[a] @ x_pred - H @ dx_pred[a])
            dB.append(dH[a] @ P_pred @ H.T + H @ dP_pred[a] @ H.T + H @ P_pred @ dH[a].T + dR[a])

        J += 0.5 * (np.linalg.slogdet(B)[1] + epsilon.T @ np.linalg.solve(B, epsilon))
        
        B_inv_eps = np.linalg.solve(B, epsilon)
        for a in range(s):
            grad_J[a] += 0.5 * (np.trace(np.linalg.solve(B, dB[a])) + 
                                2 * dEps[a].T @ B_inv_eps - 
                                B_inv_eps.T @ dB[a] @ B_inv_eps)

        K = P_pred @ H.T @ np.linalg.inv(B)
        x_filt = x_pred + K @ epsilon
        P_filt = (np.eye(n) - K @ H) @ P_pred
        
        dP_filt = []
        for a in range(s):
            dM = dP_pred[a] @ H.T + P_pred @ dH[a].T
            dK = (dM - K @ dB[a]) @ np.linalg.inv(B)
            dx_filt[a] = dx_pred[a] + dK @ epsilon + K @ dEps[a]
            dP_filt.append((np.eye(n) - K @ H) @ dP_pred[a] - (dK @ H + K @ dH[a]) @ P_pred)

    return J, grad_J

# ======================================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================================

def predict_output(theta, Y_obs):
    F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices(theta)
    x_filt = x0.copy()
    P_filt = P0.copy()
    Y_hat = []
    
    for k in range(N):
        u = get_input_signal(k)
        if k > 0:
            x_pred = F @ x_filt + Psi @ u
            P_pred = F @ P_filt @ F.T + Gamma @ Q @ Gamma.T
        else:
            x_pred = x0; P_pred = P0
            
        y_k = Y_obs[k]
        B = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(B)
        x_filt = x_pred + K @ (y_k - H @ x_pred)
        P_filt = (np.eye(n) - K @ H) @ P_pred
        Y_hat.append(H @ x_filt)
        7
    return np.array(Y_hat)

def verify_gradients(Y_obs, U_inputs):
    if not JAX_AVAILABLE: return

    print("\nПроверка градиентов:")
    theta_test = TRUE_THETA + 0.1
    
    # 1. Аналитический расчет
    J_manual, grad_manual = log_likelihood_and_gradient(theta_test, Y_obs, U_inputs)
    
    # 2. Автоматический расчет через JAX
    jax_obj_func = create_jax_optimization_wrapper(Y_obs, U_inputs)
    J_auto, grad_auto = jax_obj_func(theta_test)
    
    print(f"{'Тип':<15} | {'J (Критерий)':<15} | {'Градиент dJ/dTheta'}")
    print("-" * 70)
    print(f"{'Manual':<15} | {J_manual:<15.6f} | {grad_manual}")
    print(f"{'AutoDiff (JAX)':<15} | {J_auto:<15.6f} | {grad_auto}")
    print("-" * 70)
    
    diff_norm = np.linalg.norm(grad_manual - grad_auto)
    if diff_norm < 1e-6:
        print("Градиент в пределах погрешности")
    else:
        print(f"Недопустимое расхождение {diff_norm:.2e}")
    print("\n")

# ======================================================================================
# ЗАПУСК
# ======================================================================================

def main():
    check_system_properties(TRUE_THETA)

    USE_JAX_OPTIMIZER = True if input("Использовать JAX? (Y/n): ") == "Y" else False
    NUM_EXPERIMENTS = 5
    estimates = []
    y_true_all = []
    y_est_all = []
    
    Y_temp, U_temp = generate_data(TRUE_THETA, N)
    verify_gradients(Y_temp, U_temp)
    
    theta_init = TRUE_THETA * 0.5 + 0.1
    
    # Режим работы
    USING_JAX = JAX_AVAILABLE and USE_JAX_OPTIMIZER
    method_name = "With AutoDiff (JAX)" if USING_JAX else "Analytical"
    print(f"Метод оптимизации: {method_name}")
    print(f"{'№':<5} | {'Theta_1':<10} | {'Theta_2':<10} | {'Статус':<10}")
    
    for i in range(NUM_EXPERIMENTS):
        Y_obs, U_inputs = generate_data(TRUE_THETA, N)
        y_true_all.append(Y_obs)
        
        if USING_JAX:
            objective_function = create_jax_optimization_wrapper(Y_obs, U_inputs)
        else:
            objective_function = lambda th: log_likelihood_and_gradient(th, Y_obs, U_inputs)
        
        # Оптимизация
        res = minimize(objective_function, theta_init, method='L-BFGS-B', jac=True, bounds=THETA_BOUNDS)
        
        theta_hat = res.x
        estimates.append(theta_hat)
        Y_hat = predict_output(theta_hat, Y_obs)
        y_est_all.append(Y_hat)
        
        print(f"{i+1:<5} | {theta_hat[0]:<10.4f} | {theta_hat[1]:<10.4f} | {'OK' if res.success else 'FAIL'}")

    theta_avg = np.mean(estimates, axis=0)
    print("-" * 70)
    print(f"{'True':<5} | {TRUE_THETA[0]:<10.4f} | {TRUE_THETA[1]:<10.4f}")
    print(f"{'Avg':<5} | {theta_avg[0]:<10.4f} | {theta_avg[1]:<10.4f}")
    
    delta_theta = np.linalg.norm(TRUE_THETA - theta_avg) / np.linalg.norm(TRUE_THETA)
    
    Y_obs_mean = np.mean(np.array(y_true_all), axis=0)
    Y_est_mean = np.mean(np.array(y_est_all), axis=0)
    delta_Y = np.linalg.norm(Y_obs_mean - Y_est_mean) / np.linalg.norm(Y_obs_mean)
    
    print("-" * 70)
    print(f"Delta Theta: {delta_theta:.6f}")
    print(f"Delta Y:     {delta_Y:.6f}")
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_all[0].flatten(), 'b-o', label='Observation')
    plt.plot(y_est_all[0].flatten(), 'r--x', label='Model')
    plt.legend(); plt.grid(True); plt.savefig('plot.png')


if __name__ == "__main__":
    main()