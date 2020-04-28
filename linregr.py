import numpy as np

class LinearRegression:
    def __init__(self, l_p_metric=2, seed=42):
         """
        :param l_p_metric: Задаёт метрику для оптимизации.
        Значение 1 соответсвует MAE, 2 — MSE.
        :param seed: radom_seed для случайной инициализации весов
         """
        # np.linalg.norm
        self.metric = lambda preds, y:  np.mean((np.linalg.norm(preds-y, ord=l_p_metric, axis=1)**l_p_metric), axis=0)
        self.seed = seed
            
        self.W = None
        self.b = None
    
    
    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            нормального распределения (np.random.normal)
            со средним 0 и стандартным отклонением 0.01
            b - вектор размерности (1, output_size)
            инициализируется нулями
        """
        np.random.seed(42)
        self.W = np.random.normal(0, 0.01, size=(input_size, output_size)) 
        self.b = np.zeros((1, output_size), dtype=float) 

    def fit(self, X, y, num_epochs=1000, lr=0.001):
        """
            Обучение модели линейной регрессии методом градиентного спуска
            @param X: размерности (num_samples, input_shape)
            @param y: размерности (num_samples, output_shape)
            @param num_epochs: количество итераций градиентного спуска
            @param lr: шаг градиентного спуска
            @return metrics: вектор значений метрики на каждом шаге градиентного
            спуска. В случае mae_metric==True вычисляется метрика MAE
            иначе MSE
        """
        self.init_weights(X.shape[1], y.shape[1])
        metrics = []
        for _ in range(num_epochs):
            preds = self.predict(X)
            w_grad = 2/X.shape[0] * X.T@(preds - y)
            #w_grad = 2 * np.mean(X.T @ (preds - y), axis=0)  
            b_grad = np.mean(2 * (preds - y), axis=0)
            self.W -= lr*w_grad
            self.b -= lr*b_grad
            metrics.append(self.metric(preds, y))
        return metrics 

    def predict(self, X):
        """
            Думаю, тут все понятно. Сделайте свои предсказания :)
        """

        return X@self.W + self.b