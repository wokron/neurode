def ode_next_euler(t, yi, step, calc_dy):
    return yi + step * calc_dy(yi, t)


def ode_next_rk2(t, yi, step, calc_dy):
    k1 = calc_dy(yi, t)
    k2 = calc_dy(yi + step * k1, t + step)
    return yi + step * (k1 + k2) / 2


def ode_next_rk4(t, yi, step, calc_dy):
    k1 = calc_dy(yi, t)
    k2 = calc_dy(yi + step * k1 / 2, t + step / 2)
    k3 = calc_dy(yi + step * k2 / 2, t + step / 2)
    k4 = calc_dy(yi + step * k3, t + step)
    return yi + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
