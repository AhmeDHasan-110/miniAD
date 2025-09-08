import math
from collections import deque

class _GradFunctions:
    @staticmethod
    def leafBackward(dummy): return

    @staticmethod
    def addBackward(grad, oper1, oper2):
        oper1._part_adjoint.append(grad)
        oper2._part_adjoint.append(grad)
    
    @staticmethod
    def subBackward(grad, oper1, oper2):
        oper1._part_adjoint.append(grad)
        oper2._part_adjoint.append(-grad)

    @staticmethod
    def mulBackward(grad, oper1, oper2):
        oper1._part_adjoint.append(oper2.value * grad)
        oper2._part_adjoint.append(oper1.value * grad)

    @staticmethod
    def divBackward(grad, oper1, oper2):
        oper1._part_adjoint.append(1 / oper2.value * grad)
        oper2._part_adjoint.append((-oper1.value / oper2.value ** 2) * grad)

    @staticmethod
    def powBackward(grad, oper1, oper2):
        oper1._part_adjoint.append(oper2.value * oper1.value ** (oper2.value - 1) * grad)
        # if base is negative, derivative is not defined (because of ln())
        if oper1.value > 0 :
            oper2._part_adjoint.append(oper1.value ** oper2.value * math.log(oper1.value) * grad)
        else :
            oper2._part_adjoint.append(float("nan"))

    @staticmethod
    def reluBackward(grad, oper):
        oper._part_adjoint.append(grad * 1 if oper.value > 0 else 0)

    @staticmethod
    def tanhBackward(grad, oper):
        oper._part_adjoint.append(grad * (1 - math.tanh(oper.value) ** 2))

class Arrow:
    def __init__(self, value:float, _prev:tuple=()):
        self.value = float(value)
        self.adjoint = 0
        self._part_adjoint = []
        self._prev = _prev
        self._backward = _GradFunctions.leafBackward
    
    def _ensure_arrow(self, other):
        return other if isinstance(other, Arrow) else Arrow(other)

    def __add__(self, other):
        other = self._ensure_arrow(other)
        result = Arrow(self.value + other.value, (self, other))
        result._backward = lambda grad : _GradFunctions.addBackward(grad, self, other)
        return result
    
    def __sub__(self, other):
        other = self._ensure_arrow(other)
        result = Arrow(self.value - other.value, (self, other))
        result._backward = lambda grad : _GradFunctions.subBackward(grad, self, other)
        return result

    def __mul__(self, other) :
        other = self._ensure_arrow(other)
        result = Arrow(self.value * other.value, (self, other))
        result._backward = lambda grad : _GradFunctions.mulBackward(grad, self, other)
        return result
    
    def __truediv__(self, other):
        other = self._ensure_arrow(other)
        result = Arrow(self.value / other.value, (self, other))
        result._backward = lambda grad : _GradFunctions.divBackward(grad, self, other)
        return result

    def __pow__(self, other):
        other = self._ensure_arrow(other)

        if self.value < 0 and not other.value.is_integer() : # check the case of a square root of negative number
            raise ValueError("Derivative undefined: negative base with non-integer exponent")
        
        result = Arrow(self.value ** other.value, (self, other))
        result._backward = lambda grad : _GradFunctions.powBackward(grad, self, other)
        return result
    
    def __neg__(self):
        return self.__mul__(self, -1)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __rpow__(self, other):
        return self.__truediv__(other)
    
    def relu(self):
        result = Arrow(max(self.value, 0), (self,))
        result._backward = lambda grad : _GradFunctions.reluBackward(grad, self)
        return result
    
    def tanh(self):
        result = Arrow(math.tanh(self.value), (self,))
        result._backward = lambda grad : _GradFunctions.tanhBackward(grad, self)
        return result


    def __topological_sort(self):
        visit, res = set(), deque()
        def dfs(node):
            if node in visit:
                return
        
            visit.add(node)
            for i in node._prev:
                dfs(i)
            res.appendleft(node)
        dfs(self)
        return res

    def backward(self):
        deq = self.__topological_sort()
        self._part_adjoint.append(1)
        while deq :
            cur = deq.popleft()
            cur.adjoint = sum(cur._part_adjoint)
            cur._backward(cur.adjoint)
            cur._part_adjoint.clear()
    
    def __repr__(self):
        return f"Arrow(data={self.value})"
