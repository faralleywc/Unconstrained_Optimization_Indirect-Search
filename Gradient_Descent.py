from sympy import *
from math import *

class gradient_descent():
    def __init__(self, function, start_point, threshold, times, step_range, gold_precision):
        self.function = sympify(function)
        self.start_point = start_point
        self.threshold = threshold
        self.times = times
        self.step_range = step_range
        self.gold_precision = gold_precision
        self.alpha = Symbol('alpha')
        self.dim = len(start_point) 

    # 将符号化表达的x1、x2......xn赋以同名的变量，用于计算过程的取用
    def symbol_variable(self):
        X = []
        for i in range(self.dim):
            vars()['x%s'% (i+1)] = Symbol('x%s'% (i+1))
            X.append(symbols('x%s'% (i+1)))       
        return X

    # 将字符串x1、x2......xn存放在符号列表中
    def symbol_save(self):
        # 以符号表达方式存储自变量
        dim = len(self.start_point)
        symbol_x = []
        for i in range(dim):
            xi = 'x' + str(i+1)
            symbol_x.append(Symbol(xi))
        return symbol_x

    # 计算导数表达式
    def cal_derivative_list(self, function):
        symbol_x = self.symbol_save()
        derivative_list = []
        for i in range(self.dim):
            derivative = diff(function, symbol_x[i])
            derivative_list.append(derivative)
        return derivative_list

    def cal_gradient(self, symbol_x, x, derivative):
        # 将x的值传递给符号保存库
        sub = {}
        for i in range(self.dim):
            sub[symbol_x[i]] = x[i]
        
        # 将各个方向的梯度计算出来，作为搜索方向组合
        direction_list = []
        for i in range(self.dim):
            direction = derivative[i].evalf(subs = sub)
            direction_list.append(direction)        
        return direction_list

    # 根据坐标值计算目标函数值
    def target_calculate(self, x):
        X = self.symbol_variable()
        y = self.function.evalf(subs = dict(zip(X, x)))
        return y

    def algorithm_run(self):
        # 调用建立好的符号列表（[x1,x2,x3,...,xn]）
        symbol_x = self.symbol_save()
        # 设置点距初值和迭代次数初值
        time = 0
        # 迭代过程中的坐标，以起始点为初值
        x = self.start_point
        # 定义初始
        gradient = float('inf')
        # 各个维度的梯度表达式
        derivative = self.cal_derivative_list(self.function)

        while (gradient > self.threshold) and (time < self.times):       
            direction_list = self.cal_gradient(symbol_x, x, derivative)
            # 更新目标函数的值，使其仅为一维最优搜索步长α的函数
            target = self.function
            for i in range(self.dim):
                target = target.subs(symbol_x[i], x[i] - self.alpha * direction_list[i])
            
            # 黄金分割法确定一维搜索步长并对x进行更新
            best_step = self.golden_ratio(target)
            x = [x[j] - i * best_step for i, j in zip(direction_list, range(self.dim))]

            # 完成一轮坐标更新后，判断是否达到终止准则(梯度终止准则)
            time += 1
            gradient_list = self.cal_gradient(symbol_x, x, derivative)
            res = 0
            for item in gradient_list:
                res += item ** 2
            gradient = sqrt(res)

        return x

    def golden_ratio(self, target):
        a = -self.step_range
        b = self.step_range
        e = self.gold_precision
        ratio = float(GoldenRatio) - 1
        G1 = a + (1 - ratio) * (b - a)
        G2 = a + (b - a) * ratio
        while True:        
            # 如果搜索区间长度在精度范围内，直接得到最优步长
            if (b - a) <= e:
                midpoint = (a + b) / 2
                best_step = midpoint
                break
            # 如果搜索区间长度在精度范围外，黄金分割缩小搜索区间
            elif (b - a) > e:
                # G2分割点更优，保留b点并使G2替代G1，构造新的搜索区间
                if target.subs(self.alpha, G1) >= target.subs(self.alpha, G2):
                    a = G1
                    G1 = G2
                    G2 = a + ratio * (b - a)
                # G1分割点更优，保留a点并使G1替代G2，构造新的搜索区间
                elif target.subs(self.alpha, G1) <= target.subs(self.alpha, G2):
                    b = G2
                    G2 = G1
                    G1 = a + (1 - ratio) * (b - a)
        
        return best_step

if __name__ == "__main__":
    # 给定原始条件
    function = 'x1 ** 2 + x2 ** 2 - x1 * x2 - 10 * x1 - 4 * x2 + 60'
    start_point = [5, 4]
    threshold = 0.01
    times = 100
    step_range = 1
    gold_precision = 1e-5

    # 得到结果
    myalgorithm = gradient_descent(function, start_point, threshold, times, step_range, gold_precision)
    best_x = myalgorithm.algorithm_run()
    best_y = myalgorithm.target_calulate(best_x)

                


            

