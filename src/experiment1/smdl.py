class SMDL:
    """
    Sequential Minimum Description Length algorithm
    """
    #def __init__(self, h, T, model, beta):
    def __init__(self, model, beta):
        """
        initialize parameters
        :param h: window size
        :param T: data length
        :param model: model instance (necessary to implement 'calc_change_score' method)
        :param beta: threshold parameter
        :return:
        """
        #self.h = h
        #self.T = T
        #self.model = model
        #self.model = model(h, T)
        self.model = model()
        self.beta = beta

    #def calc_change_score(self, x, mu_max, sigma_min):
    #def calc_change_score(self, x, **kwargs):
    def calc_change_score(self, x, h, **kwargs):
        """
        calculate change score using specified model
        :param x: data
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change score
        """
        # calculate change score
        #change_score = self.model.calc_change_score(x, mu_max, sigma_min)
        #change_score = self.model.calc_change_score(x, **kwargs)
        change_score = self.model.calc_change_score(x, h, **kwargs)
        #if change_score > self.beta:
        #    print('alarm')
        return change_score