import numpy as np
from scipy.optimize import minimize
import time

# --- БЛОК 1: МОДЕЛЬ СИСТЕМЫ (ЗАПОЛНИТЕ СВОЙ ВАРИАНТ ЗДЕСЬ) ---
class SystemModel:
    def __init__(self, theta):
        """
        Здесь задаются матрицы F, Psi, Gamma, H, Q, R и начальные условия
        в зависимости от вектора параметров theta.
        """
        self.theta = theta
        
        # Пример размерностей (n=2, m=1). ЗАМЕНИТЕ НА СВОИ!
        n, m, r, p = 2, 1, 1, 2
        
        # --- 1.1 Матрицы системы [cite: 96-102] ---
        # x(k+1) = F*x(k) + Psi*u(k) + Gamma*w(k)
        # y(k+1) = H*x(k+1) + v(k+1)
        
        # ПРИМЕР: theta[0] влияет на F[0,0], theta[1] влияет на H[0,0]
        self.F = np.eye(n)
        self.F[0, 0] = theta[0]  # Параметр 1
        
        self.Psi = np.array([[1.0], [0.0]]) # Вход
        self.Gamma = np.eye(n)
        
        self.H = np.zeros((m, n))
        self.H[0, 0] = theta[1]  # Параметр 2
        
        self.Q = np.eye(p) * 0.1  # Шум системы
        self.R = np.eye(m) * 0.5  # Шум измерений
        
        self.x0 = np.zeros(n)
        self.P0 = np.eye(n)

    def get_derivatives(self, param_idx):
        """
        Возвращает производные матриц по параметру theta[param_idx].
        Это необходимо для I уровня сложности (аналитический градиент).
        [cite: 235]
        """
        n, m = self.F.shape[0], self.H.shape[0]
        p = self.Q.shape[0]
        
        # Инициализируем нулями
        dF = np.zeros_like(self.F)
        dPsi = np.zeros_like(self.Psi)
        dGamma = np.zeros_like(self.Gamma)
        dH = np.zeros_like(self.H)
        dQ = np.zeros_like(self.Q)
        dR = np.zeros_like(self.R)
        dP0 = np.zeros_like(self.P0)
        dx0 = np.zeros_like(self.x0)
        
        # --- 1.2 Производные (ЗАПОЛНИТЕ САМИ) ---
        # Если F[0,0] = theta[0], то производная dF/dtheta[0] будет 1 в позиции [0,0]
        
        if param_idx == 0: # Производная по первому параметру (F[0,0])
            dF[0, 0] = 1.0
            
        elif param_idx == 1: # Производная по второму параметру (H[0,0])
            dH[0, 0] = 1.0
            
        return dF, dPsi, dGamma, dH, dQ, dR, dx0, dP0

# --- БЛОК 2: ФИЛЬТР КАЛМАНА И ГРАДИЕНТЫ (НЕ МЕНЯТЬ) ---
def kalman_filter_with_gradients(theta, U, Y):
    """
    Реализует алгоритм вычисления критерия и его градиента 
    для I уровня сложности [cite: 193-206, 233-256].
    """
    model = SystemModel(theta)
    F, Psi, Gamma, H, Q, R = model.F, model.Psi, model.Gamma, model.H, model.Q, model.R
    x0, P0 = model.x0, model.P0
    
    N = len(U)
    m = Y.shape[1]
    n_params = len(theta)
    
    # Инициализация переменных фильтра
    x_pred = x0.copy()
    P_pred = P0.copy()
    
    # Инициализация переменных для градиентов (чувствительности)
    # dx_pred_dtheta[i] хранит матрицу/вектор производной по i-му параметру
    dx_pred_dtheta = [np.zeros_like(x0) for _ in range(n_params)]
    dP_pred_dtheta = [np.zeros_like(P0) for _ in range(n_params)]
    
    # Критерий J (Хи квадрат) [cite: 159]
    # N*m*v/2 * ln(2pi). v=1.
    J = (N * m / 2.0) * np.log(2 * np.pi)
    
    # Вектор градиента критерия [cite: 255]
    grad_J = np.zeros(n_params)
    
    for k in range(N):
        u_k = U[k]
        y_real = Y[k]
        
        # 1. Экстраполяция (Time Update) [cite: 163-166]
        x_prior = F @ x_pred + Psi @ u_k
        P_prior = F @ P_pred @ F.T + Gamma @ Q @ Gamma.T
        
        # Производные экстраполяции [cite: 241, 248]
        dx_prior_list = []
        dP_prior_list = []
        
        for i in range(n_params):
            dF, dPsi, dGamma, dH_m, dQ_m, dR_m, _, _ = model.get_derivatives(i)
            
            # d(x_prior)/d(theta)
            dx_p = dF @ x_pred + F @ dx_pred_dtheta[i] + dPsi @ u_k
            dx_prior_list.append(dx_p)
            
            # d(P_prior)/d(theta)
            dP_p = (dF @ P_pred @ F.T + F @ dP_pred_dtheta[i] @ F.T + 
                    F @ P_pred @ dF.T + 
                    dGamma @ Q @ Gamma.T + Gamma @ dQ_m @ Gamma.T + Gamma @ Q @ dGamma.T)
            dP_prior_list.append(dP_p)

        # 2. Инновация
        B = H @ P_prior @ H.T + R  # [cite: 169]
        epsilon = y_real - H @ x_prior # [cite: 167]
        
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return 1e10, np.ones(n_params)*1e10 # Защита от вырождения
            
        K = P_prior @ H.T @ B_inv # [cite: 174]
        
        # Обновление критерия J [cite: 159, 204]
        term1 = np.log(np.linalg.det(B))
        term2 = epsilon.T @ B_inv @ epsilon
        J += 0.5 * (term1 + term2)
        
        # 3. Обновление (Measurement Update) [cite: 175-176]
        x_post = x_prior + K @ epsilon
        I_mat = np.eye(len(x0))
        P_post = (I_mat - K @ H) @ P_prior
        
        # --- Расчет ГРАДИЕНТОВ на шаге k [cite: 242-255] ---
        for i in range(n_params):
            dF, _, _, dH_m, _, dR_m, _, _ = model.get_derivatives(i)
            dx_p = dx_prior_list[i]
            dP_p = dP_prior_list[i]
            
            # d(epsilon)/d(theta) [cite: 248]
            d_eps = -dH_m @ x_prior - H @ dx_p
            
            # d(B)/d(theta) [cite: 242]
            dB = dH_m @ P_prior @ H.T + H @ dP_p @ H.T + H @ P_prior @ dH_m.T + dR_m
            
            # d(K)/d(theta) [cite: 242]
            # K = M * B^-1, где M = P_prior * H^T
            M = P_prior @ H.T
            dM = dP_p @ H.T + P_prior @ dH_m.T
            # d(B^-1) = -B^-1 * dB * B^-1
            dK = (dM - K @ dB) @ B_inv
            
            # Накопление градиента критерия [cite: 226, 255]
            # Часть следа: 0.5 * tr(B^-1 * dB)
            grad_trace = 0.5 * np.trace(B_inv @ dB)
            
            # Часть невязки: производная квадратичной формы e^T * B^-1 * e
            # = d_eps^T * B^-1 * eps + eps^T * B^-1 * d_eps - eps^T * B^-1 * dB * B^-1 * eps
            # (первые два слагаемых равны, т.к. скаляр)
            term_grad_quad = (d_eps.T @ B_inv @ epsilon) - 0.5 * (epsilon.T @ B_inv @ dB @ B_inv @ epsilon)
            
            grad_J[i] += grad_trace + term_grad_quad
            
            # d(x_post)/d(theta) для следующего шага [cite: 248]
            dx_pred_dtheta[i] = dx_p + dK @ epsilon + K @ d_eps
            
            # d(P_post)/d(theta) для следующего шага [cite: 242]
            # P_post = (I - KH)P_prior
            # dP_post = - (dK*H + K*dH)*P_prior + (I-KH)*dP_prior
            dP_pred_dtheta[i] = (I_mat - K @ H) @ dP_p - (dK @ H + K @ dH_m) @ P_prior

        # Переход к следующему шагу
        x_pred = x_post
        P_pred = P_post

    return J, grad_J

# --- БЛОК 3: ЗАПУСК ---
if __name__ == "__main__":
    # 1. Генерируем "истинные" данные для теста
    print("Генерация данных...")
    true_theta = [0.8, 1.2] # Истинные значения параметров
    
    # Создаем фиктивный входной сигнал (например, единичный скачок)
    N_samples = 30
    U_data = [np.array([1.0]) for _ in range(N_samples)]
    
    # Генерируем Y используя модель (без градиентов)
    model_true = SystemModel(true_theta)
    Y_data = []
    x = model_true.x0
    for k in range(N_samples):
        w = np.random.multivariate_normal(np.zeros(2), model_true.Q)
        v = np.random.multivariate_normal(np.zeros(1), model_true.R)
        x_next = model_true.F @ x + model_true.Psi @ U_data[k] + model_true.Gamma @ w
        y_meas = model_true.H @ x_next + v
        Y_data.append(y_meas)
        x = x_next
    Y_data = np.array(Y_data)

    # 2. Запуск оптимизации
    print("Запуск идентификации (Метод I уровня с аналитическим градиентом)...")
    
    start_time = time.time()
    
    # Начальное приближение
    theta_init = [0.5, 0.5]
    
    # Обертка для minimize
    res = minimize(
        fun=kalman_filter_with_gradients, # Функция возвращает (J, grad)
        x0=theta_init,
        args=(U_data, Y_data),
        method='BFGS', # Или 'SLSQP'. BFGS отлично работает с аналитическим градиентом
        jac=True,      # Указываем, что функция возвращает градиент!
        options={'disp': True}
    )
    
    print("\n--- Результат ---")
    print(f"Истинные параметры: {true_theta}")
    print(f"Найденные параметры: {res.x}")
    print(f"Время выполнения: {time.time() - start_time:.4f} сек")
    
    # Ошибка
    error = np.linalg.norm(np.array(true_theta) - res.x) / np.linalg.norm(true_theta)
    print(f"Относительная ошибка: {error:.4f}")