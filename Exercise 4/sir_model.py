
from scipy.integrate import solve_ivp

def mu(b, I, mu0, mu1):
    """Recovery rate.
    
    """
    # recovery rate, depends on mu0, mu1, b
    mu = mu0 + (mu1 - mu0) * (b/(I+b))
    return mu

def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.
    """
    return beta / (d + nu + mu1)

def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.
    """
    c0 = b**2 * d * A
    c1 = b * ((mu0-mu1+2*d) * A + (beta-nu)*b*d)
    c2 = (mu1-mu0)*b*nu + 2*b*d*(beta-nu)+d*A
    c3 = d*(beta-nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res
    

def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.
    
    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """
    S,I,R = y[:]
    m = mu(b, I, mu0, mu1)
    
    infected = beta * S * I / (S + I + R)
    recovered = m * I

    dSdt = A - d * S - infected
    dIdt = - (d + nu) * I - recovered + infected
    dRdt = recovered - d * R
    
    return [dSdt, dIdt, dRdt]


def simulate_and_plot(SIM0, mu0, mu1, beta, A, d, nu, b, time, ax, rtol=1e-8, atol=1e-8):
    sol = solve_ivp(model, t_span=[time[0],time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b), method='DOP853', rtol=rtol, atol=atol)

    # ax.plot(sol.y[0], sol.y[1], sol.y[2], 'r-');
    ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time);
    # ax.scatter(sol.y[0], sol.y[1], sol.y[2], c=time);

    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.set_zlabel("R")

    ax.set_title(f"{SIM0}, b={b:.3f}") 