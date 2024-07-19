

#Стохастический градиентный спуск. Суть его в том, что веса изменяются при каждой итерации,
# а не раз в период, как,например, в пакетном.
class Stochastic_gradient_descent(object):

    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        #alpha отвечает за уменьшение шагов градиентного спуска, без нее может не получиться найти локальный минимум
        self.alpha = alpha
    # Используется для зануления конкретных весов нейросети. Это может помочь при переобучении
    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):

        for p in self.parameters:
            p.data -= p.grad.data * self.alpha

            if (zero):
                p.grad.data *= 0