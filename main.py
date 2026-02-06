import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ======================================================================================
# НАСТРОЙКИ ВАРИАНТА (ЗДЕСЬ НУЖНО ВНЕСТИ ДАННЫЕ СВОЕГО ВАРИАНТА)
# ======================================================================================

# Размерности
N = 30      # Длина выборки (количество тактов времени)
n = 2       # Размерность вектора состояния x
m = 1       # Размерность вектора измерений y
r = 1       # Размерность вектора управления u
p = 2       # Размерность шума системы w (часто совпадает с n)

# Истинные значения параметров (theta*)
# ПРИМЕР: Допустим, у нас 2 неизвестных параметра.
# theta[0] - элемент матрицы F, theta[1] - элемент матрицы Psi
TRUE_THETA = np.array([0.8, 0.5]) 

# Входной сигнал u(t)
# ПРИМЕР: Единичный скачок или случайный сигнал
def get_input_signal(k):
    return np.array([1.0]) # Постоянный вход u=1

# ======================================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ МАТРИЦ (ЗАВИСИМОСТЬ ОТ ПАРАМЕТРОВ)
# ======================================================================================

def get_system_matrices(theta):
    """
    Возвращает матрицы F, Psi, Gamma, H, Q, R, x0, P0 для заданного вектора параметров theta.
    Здесь нужно прописать структуру твоего варианта.
    """
    th1, th2 = theta[0], theta[1]
    
    # ПРИМЕР:
    # x(k+1) = [[th1, 0], [0, 0.9]] * x(k) + [[th2], [1]] * u(k) + w(k)
    F = np.array([[th1, 0.0], 
                  [0.0, 0.9]])
    
    Psi = np.array([[th2], 
                    [1.0]])
    
    Gamma = np.eye(n) # Единичная матрица
    
    H = np.array([[1.0, 0.0]]) # Измеряем только первый элемент состояния
    
    Q = np.eye(p) * 0.1  # Ковариация шума системы
    R = np.eye(m) * 0.1  # Ковариация шума измерений
    
    x0_mean = np.zeros(n)
    P0 = np.eye(n) * 1.0
    
    return F, Psi, Gamma, H, Q, R, x0_mean, P0

def get_matrix_derivatives(theta):
    """
    Возвращает производные матриц по каждому параметру theta.
    dF_dtheta[alpha] - это матрица dF/dtheta_alpha
    """
    # Количество параметров
    s = len(theta)
    
    # Инициализация списков производных (по умолчанию нули)
    dF = [np.zeros((n, n)) for _ in range(s)]
    dPsi = [np.zeros((n, r)) for _ in range(s)]
    dGamma = [np.zeros((n, p)) for _ in range(s)]
    dH = [np.zeros((m, n)) for _ in range(s)]
    dQ = [np.zeros((p, p)) for _ in range(s)]
    dR = [np.zeros((m, m)) for _ in range(s)]
    dx0 = [np.zeros(n) for _ in range(s)]
    dP0 = [np.zeros((n, n)) for _ in range(s)]
    
    # ПРИМЕР (согласованный с get_system_matrices):
    # F = [[th1, 0], [0, 0.9]] => dF/dth1 = [[1,0],[0,0]], dF/dth2 = 0
    dF[0][0, 0] = 1.0
    
    # Psi = [[th2], [1]] => dPsi/dth1 = 0, dPsi/dth2 = [[1],[0]]
    dPsi[1][0, 0] = 1.0
    
    return dF, dPsi, dGamma, dH, dQ, dR, dx0, dP0

# ======================================================================================
# ЯДРО: ГЕНЕРАЦИЯ ДАННЫХ И ФИЛЬТР КАЛМАНА С ГРАДИЕНТОМ
# ======================================================================================

def generate_data(theta_true, N_samples):
    """Генерация одной реализации эксперимента"""
    F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices(theta_true)
    
    # Инициализация
    x = np.random.multivariate_normal(x0, P0)
    Y_obs = []
    
    for k in range(N_samples):
        u = get_input_signal(k)
        
        # Шум системы и измерения
        w = np.random.multivariate_normal(np.zeros(p), Q)
        v = np.random.multivariate_normal(np.zeros(m), R)
        
        # Измерение
        y = H @ x + v
        Y_obs.append(y)
        
        # Динамика состояния
        x = F @ x + Psi @ u + Gamma @ w
        
    return np.array(Y_obs)

def log_likelihood_and_gradient(theta, Y_observations):
    """
    Вычисляет критерий J (отрицательный логарифм правдоподобия) 
    и его градиент dJ/dtheta.
    Реализует алгоритм I уровня сложности (стр. 10-11 методички).
    """
    try:
        F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices(theta)
        dF, dPsi, dGamma, dH, dQ, dR, dx0, dP0 = get_matrix_derivatives(theta)
    except:
        # Защита от выхода параметров за недопустимые границы (если есть log/sqrt)
        return 1e10, np.zeros(len(theta))

    s = len(theta)
    
    # Инициализация переменных фильтра
    x_pred = x0.copy()          # x(k+1|k)
    P_pred = P0.copy()          # P(k+1|k)
    x_filt = x0.copy()          # x(k|k)
    P_filt = P0.copy()          # P(k|k)
    
    # Инициализация производных фильтра (чувствительность)
    # dx_pred[alpha] = d(x_pred)/dtheta_alpha
    dx_pred = [dx0[i].copy() for i in range(s)]
    dP_pred = [dP0[i].copy() for i in range(s)]
    dx_filt = [dx0[i].copy() for i in range(s)]
    
    J = 0.0 # Значение критерия
    grad_J = np.zeros(s) # Градиент
    
    # Константа из формулы (8)
    # N*m*v*ln(2pi). Здесь v=1 (один прогон в итерации оптимизатора), считаем сумму по k
    J += 0.5 * N * m * np.log(2 * np.pi)
    
    for k in range(N):
        u = get_input_signal(k)
        y_obs = Y_observations[k]
        
        # --- ШАГ 1: Прогноз состояния x(k+1|k) и ковариации P(k+1|k) ---
        # Формулы (9), (10)
        # При k=0, x_pred уже инициализирован x0. Для k>0 вычисляем:
        if k > 0:
            x_pred = F @ x_filt + Psi @ u
            P_pred = F @ P_filt @ F.T + Gamma @ Q @ Gamma.T
            
            # Производные прогноза (Стр 10, шаг 4 и 10)
            for a in range(s):
                dx_pred[a] = dF[a] @ x_filt + F @ dx_filt[a] + dPsi[a] @ u
                
                term1 = dF[a] @ P_filt @ F.T
                term2 = F @ dP_filt[a] @ F.T # Внимание: dP_filt нужен с прошлого шага
                term3 = F @ P_filt @ dF[a].T
                term4 = dGamma[a] @ Q @ Gamma.T
                term5 = Gamma @ dQ[a] @ Gamma.T
                term6 = Gamma @ Q @ dGamma[a].T
                dP_pred[a] = term1 + term2 + term3 + term4 + term5 + term6

        # --- ШАГ 2: Инновация (невязка) ---
        # Формулы (11), (12)
        epsilon = y_obs - H @ x_pred
        B = H @ P_pred @ H.T + R
        
        # Инверсия B (для устойчивости используем solve)
        # B_inv = np.linalg.inv(B)
        
        # --- ШАГ 3: Производные B и epsilon ---
        dB = []
        dEps = []
        
        for a in range(s):
            # d_epsilon / d_theta (Стр 10)
            de = -dH[a] @ x_pred - H @ dx_pred[a]
            dEps.append(de)
            
            # d_B / d_theta (Стр 10)
            db = dH[a] @ P_pred @ H.T + H @ dP_pred[a] @ H.T + H @ P_pred @ dH[a].T + dR[a]
            dB.append(db)

        # --- ШАГ 4: Накопление критерия J ---
        # J += 0.5 * (ln det B + eps^T * B^-1 * eps)
        log_det_B = np.linalg.slogdet(B)[1]
        term_quad = epsilon.T @ np.linalg.solve(B, epsilon)
        J += 0.5 * (log_det_B + term_quad)
        
        # --- ШАГ 5: Накопление Градиента ---
        # Стр 11, шаг 14
        B_inv_eps = np.linalg.solve(B, epsilon) # B^-1 * eps
        
        for a in range(s):
            B_inv_dB = np.linalg.solve(B, dB[a]) # B^-1 * dB
            
            # След: tr(B^-1 * dB)
            trace_term = np.trace(B_inv_dB)
            
            # Квадратичная часть градиента
            # d(eps^T B^-1 eps) = 2*dEps^T B^-1 eps - eps^T B^-1 dB B^-1 eps
            term1_grad = 2 * dEps[a].T @ B_inv_eps
            term2_grad = - B_inv_eps.T @ dB[a] @ B_inv_eps
            
            delta_alpha = term1_grad + term2_grad
            
            grad_J[a] += 0.5 * (trace_term + delta_alpha)

        # --- ШАГ 6: Обновление (Фильтрация) ---
        # K = P_pred * H^T * B^-1
        K = P_pred @ H.T @ np.linalg.inv(B)
        
        # x(k+1|k+1) = x_pred + K * epsilon
        x_filt = x_pred + K @ epsilon
        
        # P(k+1|k+1) = (I - KH) P_pred
        I = np.eye(n)
        P_filt = (I - K @ H) @ P_pred
        
        # --- ШАГ 7: Производные обновления (для следующего шага) ---
        dP_filt = [] # Нужно для следующей итерации прогноза
        
        for a in range(s):
            # dK / d_theta
            # dK = (dP_pred H^T + P_pred dH^T - P_pred H^T B^-1 dB) B^-1
            # Упростим: K = M * B^-1, где M = P_pred H^T
            # dK = dM * B^-1 + M * d(B^-1) = dM B^-1 - M B^-1 dB B^-1
            # dK = (dM - K dB) B^-1
            dM = dP_pred[a] @ H.T + P_pred @ dH[a].T
            dK = (dM - K @ dB[a]) @ np.linalg.inv(B)
            
            # dx_filt / d_theta = dx_pred + dK eps + K dEps
            dxf = dx_pred[a] + dK @ epsilon + K @ dEps[a]
            dx_filt[a] = dxf
            
            # dP_filt / d_theta
            # dP_filt = (I - KH) dP_pred - (dK H + K dH) P_pred
            dpf = (I - K @ H) @ dP_pred[a] - (dK @ H + K @ dH[a]) @ P_pred
            dP_filt.append(dpf)

    return J, grad_J

# ======================================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================================

def predict_output(theta, Y_obs):
    """
    Вычисляет прогнозируемый выход Y_hat (одношаговый прогноз)
    для расчета ошибки в пространстве откликов.
    """
    F, Psi, Gamma, H, Q, R, x0, P0 = get_system_matrices(theta)
    
    x_pred = x0.copy()
    x_filt = x0.copy()
    P_filt = P0.copy()
    
    Y_hat = []
    
    for k in range(N):
        u = get_input_signal(k)
        
        if k > 0:
            x_pred = F @ x_filt + Psi @ u
            P_pred = F @ P_filt @ F.T + Gamma @ Q @ Gamma.T
        else:
            P_pred = P0
            
        # Оценка y(k|k-1) или y(k|k). В задании: y_hat(t_k+1 | t_k+1) = H * x_hat(t_k+1 | t_k+1)
        # Нам нужно сначала обновить состояние
        y_k = Y_obs[k]
        epsilon = y_k - H @ x_pred
        B = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(B)
        
        x_filt = x_pred + K @ epsilon
        P_filt = (np.eye(n) - K @ H) @ P_pred
        
        y_hat_val = H @ x_filt
        Y_hat.append(y_hat_val)
        
    return np.array(Y_hat)

# ======================================================================================
# ОСНОВНОЙ ЦИКЛ
# ======================================================================================

def main():
    print("=== ЛАБОРАТОРНАЯ РАБОТА №1 (I УРОВЕНЬ) ===")
    print("Активная параметрическая идентификация (Метод Максимального Правдоподобия с Градиентом)")
    print(f"Истинные параметры: {TRUE_THETA}")
    print("-" * 60)
    
    # Количество запусков
    NUM_EXPERIMENTS = 5
    
    estimates = []
    y_true_all = []
    y_est_all = []
    
    # Начальное приближение (должно быть не слишком далеко от истины для сходимости)
    theta_init = TRUE_THETA * 0.5 + 0.1 # Смещение для проверки работы
    
    print(f"{'№':<5} | {'Theta_1':<10} | {'Theta_2':<10} | {'Статус':<10}")
    
    for i in range(NUM_EXPERIMENTS):
        # 1. Генерация данных
        Y_obs = generate_data(TRUE_THETA, N)
        y_true_all.append(Y_obs)
        
        # 2. Оптимизация
        # Функция цели для минимизатора
        fun = lambda th: log_likelihood_and_gradient(th, Y_obs)
        
        # Запуск минимизации (BFGS использует градиент, переданный в fun через jac=True)
        res = minimize(fun, theta_init, method='BFGS', jac=True, options={'disp': False})
        
        theta_hat = res.x
        estimates.append(theta_hat)
        
        # 3. Сохранение отклика модели
        Y_hat = predict_output(theta_hat, Y_obs)
        y_est_all.append(Y_hat)
        
        print(f"{i+1:<5} | {theta_hat[0]:<10.4f} | {theta_hat[1]:<10.4f} | {'OK' if res.success else 'FAIL'}")

    estimates = np.array(estimates)
    theta_avg = np.mean(estimates, axis=0)
    
    print("-" * 60)
    print(f"{'Sred':<5} | {theta_avg[0]:<10.4f} | {theta_avg[1]:<10.4f}")
    
    # --- Расчет ошибок ---
    
    # 1. Ошибка параметров
    delta_theta = np.linalg.norm(TRUE_THETA - theta_avg) / np.linalg.norm(TRUE_THETA)
    
    # 2. Ошибка откликов
    # Y_avg (усредненное по экспериментам)
    Y_obs_mean = np.mean(np.array(y_true_all), axis=0)
    Y_est_mean = np.mean(np.array(y_est_all), axis=0)
    
    # Норма матрицы наблюдений (растянутой в вектор) или средняя норма по времени
    norm_Y = np.linalg.norm(Y_obs_mean)
    norm_diff = np.linalg.norm(Y_obs_mean - Y_est_mean)
    
    delta_Y = norm_diff / norm_Y
    
    print("=" * 60)
    print(f"Относительная ошибка по параметрам (delta_theta): {delta_theta:.6f} ({(delta_theta*100):.2f}%)")
    print(f"Относительная ошибка по отклику (delta_Y):       {delta_Y:.6f} ({(delta_Y*100):.2f}%)")
    
    # Визуализация первого эксперимента (для проверки)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_all[0].flatten(), 'b-o', label='Наблюдения (Exp 1)')
    plt.plot(y_est_all[0].flatten(), 'r--x', label='Модель (Оценка)')
    plt.title('Сравнение выхода системы и модели (Эксперимент 1)')
    plt.xlabel('Время k')
    plt.ylabel('Выход y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()