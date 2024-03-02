from neurode.ode import ODE
from neurode.calc import Derivation as derivate, Placeholder as var, Equations, min, max
from neurode.ode_next import ode_next_euler, ode_next_rk2, ode_next_rk4

# time varible `t`
t = var("t")
