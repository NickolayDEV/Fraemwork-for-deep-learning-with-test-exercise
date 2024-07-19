#Функции активации добавляют в нейронную сеть нелинейность
#Обе из представленных здесь функций обычно концентрируют данные около нуля
#Если сказть совсем грубо, TAnh подойдет для более длинных градиентнеых шагов, а sigmoid - для более сбалансированных

class Layer(object):

    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()