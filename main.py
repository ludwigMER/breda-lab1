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
    print("ВНИМАНИЕ: Библиотека JAX не найдена. Сравнение с авто-дифференцированием будет пропущено.")
    print("Для установки: pip install jax jaxlib")

# ======================================================================================
# НАСТРОЙКИ ВАРИАНТА
# ======================================================================================

N = 30      # Длина выборки
n = 2       # Размерность x
m = 1       # Размерность y
r = 1       # Размерность u
p = 2       # Размерность w

TRUE_THETA = np.array([0.8, 0.5]) 

def get_input_signal(k):
    return np.array([1.0])

# ======================================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ МАТРИЦ (NUMPY - ДЛЯ РУЧНОГО ГРАДИЕНТА)
# ======================================================================================

def get_system_matrices(theta):
    """Версия для ручного градиента (использует numpy)"""
    th1, th2 = theta[0], theta[1]
    
    F = np.array([[th1, 0.0], [0.0, 0.9]])
    Psi = np.array([[th2], [1.0]])
    Gamma = np.eye(n)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(p) * 0.1
    R = np.eye(m) * 0.1
    x0_mean = np.zeros(n)
    P0 = np.eye(n) * 1.0
    
    return F, Psi, Gamma, H, Q, R, x0_mean, P0

def get_matrix_derivatives(theta):
    """Аналитические производные матриц (для I уровня сложности)"""
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

# ======================================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ МАТРИЦ (JAX - ДЛЯ АВТО-ДИФФЕРЕНЦИРОВАНИЯ)
# ======================================================================================

if JAX_AVAILABLE:
    def get_system_matrices_jax(theta):
        """
        То же самое, что get_system_matrices, но используем jnp.array.
        JAX должен видеть математику, чтобы взять производную.
        """
        th1, th2 = theta[0], theta[1]
        
        F = jnp.array([[th1, 0.0], [0.0, 0.9]])
        Psi = jnp.array([[th2], [1.0]])
        Gamma = jnp.eye(n)
        H = jnp.array([[1.0, 0.0]])
        Q = jnp.eye(p) * 0.1
        R = jnp.eye(m) * 0.1
        x0_mean = jnp.zeros(n)
        P0 = jnp.eye(n) * 1.0
        
        return F, Psi, Gamma, H, Q, R, x0_mean, P0

    def calculate_loss_jax(theta, Y_observations_jax, U_inputs_jax):
        """
        Чистая функция расчета критерия J (без ручного градиента).
        JAX автоматически построит граф вычислений для этой функции.
        """
        F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices_jax(theta)
        
        # JAX требует аккуратности с циклами, но для gradient check 
        # простой python loop допустим, если массивы не изменяются in-place.
        
        x_pred = x0
        P_pred = P0
        x_filt = x0
        P_filt = P0
        
        J = 0.0
        # Константа
        J += 0.5 * N * m * jnp.log(2 * jnp.pi)
        
        # Проходим по всем измерениям
        # Примечание: В JAX лучше использовать jax.lax.scan для скорости, 
        # но для проверки градиента обычный цикл понятнее.
        for k in range(N):
            u = U_inputs_jax[k]
            y_obs = Y_observations_jax[k]
            
            # 1. Прогноз (если k > 0)
            # В Python `if` внутри JIT может быть проблемой, но здесь мы не используем @jit
            # для внешней функции, так что Python control flow сработает.
            if k > 0:
                 x_pred = F @ x_filt + Psi @ u
                 P_pred = F @ P_filt @ F.T + Gamma @ Q @ Gamma.T
            else:
                 # Для k=0 x_pred = x0 (уже задано)
                 pass

            # 2. Инновация
            epsilon = y_obs - H @ x_pred
            B = H @ P_pred @ H.T + R
            
            # 3. Накопление критерия
            # slogdet возвращает (sign, logdet), нам нужен logdet
            _, log_det_B = jnp.linalg.slogdet(B)
            
            # eps.T * B^-1 * eps
            # jnp.linalg.solve(a, b) решает ax = b -> x = a^-1 b
            term_quad = epsilon.T @ jnp.linalg.solve(B, epsilon)
            
            J += 0.5 * (log_det_B + term_quad)
            
            # 4. Обновление (Фильтрация)
            K = P_pred @ H.T @ jnp.linalg.inv(B)
            x_filt = x_pred + K @ epsilon
            P_filt = (jnp.eye(n) - K @ H) @ P_pred
            
        return J

# ======================================================================================
# ОСНОВНАЯ ЛОГИКА (РУЧНОЙ МЕТОД)
# ======================================================================================

def generate_data(theta_true, N_samples):
    F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices(theta_true)
    x = np.random.multivariate_normal(x0, P0)
    Y_obs = []
    U_inputs = [] # Сохраняем входы для чистоты эксперимента
    
    for k in range(N_samples):
        u = get_input_signal(k)
        U_inputs.append(u)
        w = np.random.multivariate_normal(np.zeros(p), Q)
        v = np.random.multivariate_normal(np.zeros(m), R)
        y = H @ x + v
        Y_obs.append(y)
        x = F @ x + Psi @ u + Gamma @ w
        
    return np.array(Y_obs), np.array(U_inputs)

def log_likelihood_and_gradient(theta, Y_observations):
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
        u = get_input_signal(k)
        y_obs = Y_observations[k]
        
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
# ВСПОМОГАТЕЛЬНЫЕ
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
        
    return np.array(Y_hat)

def verify_gradients(Y_obs, U_inputs):
    """
    Функция сверки градиентов: Ручной vs AutoDiff (JAX)
    """
    if not JAX_AVAILABLE:
        return

    print("\n=== ПРОВЕРКА ГРАДИЕНТОВ (JAX vs ANALYTICAL) ===")
    
    # Точка проверки (случайная или истинная)
    theta_test = TRUE_THETA + 0.1
    print(f"Точка проверки theta: {theta_test}")
    
    # 1. Аналитический расчет
    J_manual, grad_manual = log_likelihood_and_gradient(theta_test, Y_obs)
    
    # 2. Автоматический расчет через JAX
    # Преобразуем данные в JAX массивы
    Y_jax = jnp.array(Y_obs)
    U_jax = jnp.array(U_inputs)
    
    # Создаем функцию, которая возвращает значение и градиент
    value_and_grad_fn = jax.value_and_grad(calculate_loss_jax)
    
    # Вычисляем
    J_auto, grad_auto = value_and_grad_fn(jnp.array(theta_test), Y_jax, U_jax)
    
    # Преобразуем обратно в numpy для вывода
    J_auto = float(J_auto)
    grad_auto = np.array(grad_auto)
    
    print(f"{'Тип':<15} | {'J (Критерий)':<15} | {'Градиент dJ/dTheta'}")
    print("-" * 65)
    print(f"{'Manual':<15} | {J_manual:<15.6f} | {grad_manual}")
    print(f"{'AutoDiff (JAX)':<15} | {J_auto:<15.6f} | {grad_auto}")
    print("-" * 65)
    
    diff_norm = np.linalg.norm(grad_manual - grad_auto)
    print(f"Разница норм градиентов: {diff_norm:.2e}")
    
    if diff_norm < 1e-6:
        print(">> РЕЗУЛЬТАТ: Аналитический градиент ВЕРЕН! ✅")
    else:
        print(">> РЕЗУЛЬТАТ: Обнаружено расхождение! Проверьте формулы производных ❌")
    print("=" * 65 + "\n")

# ======================================================================================
# MAIN
# ======================================================================================

def main():
    print("=== ЛАБОРАТОРНАЯ РАБОТА №1 + JAX CHECK ===")
    
    NUM_EXPERIMENTS = 5
    estimates = []
    y_true_all = []
    y_est_all = []
    
    # Для проверки градиента генерируем один набор данных
    Y_temp, U_temp = generate_data(TRUE_THETA, N)
    verify_gradients(Y_temp, U_temp) # <-- ВЫЗОВ ПРОВЕРКИ
    
    theta_init = TRUE_THETA * 0.5 + 0.1
    print(f"{'№':<5} | {'Theta_1':<10} | {'Theta_2':<10} | {'Статус':<10}")
    
    for i in range(NUM_EXPERIMENTS):
        Y_obs, _ = generate_data(TRUE_THETA, N)
        y_true_all.append(Y_obs)
        
        fun = lambda th: log_likelihood_and_gradient(th, Y_obs)
        res = minimize(fun, theta_init, method='BFGS', jac=True, options={'disp': False})
        
        theta_hat = res.x
        estimates.append(theta_hat)
        Y_hat = predict_output(theta_hat, Y_obs)
        y_est_all.append(Y_hat)
        
        print(f"{i+1:<5} | {theta_hat[0]:<10.4f} | {theta_hat[1]:<10.4f} | {'OK' if res.success else 'FAIL'}")

    theta_avg = np.mean(estimates, axis=0)
    print("-" * 60)
    print(f"{'Sred':<5} | {theta_avg[0]:<10.4f} | {theta_avg[1]:<10.4f}")
    
    delta_theta = np.linalg.norm(TRUE_THETA - theta_avg) / np.linalg.norm(TRUE_THETA)
    
    Y_obs_mean = np.mean(np.array(y_true_all), axis=0)
    Y_est_mean = np.mean(np.array(y_est_all), axis=0)
    delta_Y = np.linalg.norm(Y_obs_mean - Y_est_mean) / np.linalg.norm(Y_obs_mean)
    
    print("=" * 60)
    print(f"Error Theta: {delta_theta:.6f}")
    print(f"Error Y:     {delta_Y:.6f}")
    
    if JAX_AVAILABLE:
        print("\n(Примечание: Оптимизация выполнялась с использованием вашего АНАЛИТИЧЕСКОГО градиента,")
        print("JAX использовался только для проверки его корректности в начале работы)")

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_all[0].flatten(), 'b-o', label='Observation')
    plt.plot(y_est_all[0].flatten(), 'r--x', label='Model')
    plt.legend(); plt.grid(True); plt.show()

if __name__ == "__main__":
    main()