import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp


def ask(prompt, cast=float, min_value=None, max_value=None):
    #Keep asking until the user types a valid number.
    while True:
        text = input(prompt).strip()
        try:
            value = cast(text)
        except ValueError:
            print("  Please enter a valid number.")
            continue
        if min_value is not None and value < min_value:
            print(f"  Value must be at least {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"  Value must be at most {max_value}.")
            continue
        return value


# -------------------------------------------------------------
# PART 1: Population growth (logistic model)  - ODE, Unit V
#   dP/dt = r * P * (1 - P/K)
#   P(t)  = K / (1 + ((K - P0)/P0) * e^(-r t))
# Inputs: starting population, maximum population, growth % per day.
# The time axis is chosen automatically (until P reaches 99% of K).
# -------------------------------------------------------------
print("=== PART 1: Population growth ===")
P0 = ask("Starting population (e.g. 10): ", float, min_value=1)
K_val = ask("Maximum population the area can support (e.g. 1000): ", float, min_value=P0 + 1)
growth_pct = ask("Growth per day when the population is small, in % (e.g. 50): ", float, min_value=0.01)
r_val = growth_pct / 100

t_s = sp.symbols("t")
P = sp.Function("P")
r, K = sp.symbols("r K", positive=True)
print("\nLogistic growth ODE solution:")
sp.pprint(sp.dsolve(sp.Eq(P(t_s).diff(t_s), r * P(t_s) * (1 - P(t_s) / K))))

# time for P to reach 99% of K, solved from the closed form
t_end = np.log(99 * (K_val - P0) / P0) / r_val
t = np.linspace(0, t_end, 300)
P_t = K_val / (1 + ((K_val - P0) / P0) * np.exp(-r_val * t))

plt.figure(figsize=(7, 4))
plt.plot(t, P_t, color="green")
plt.axhline(K_val, linestyle="--", color="gray", label="Maximum population K")
plt.title("Logistic Population Growth")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("1_logistic_growth.png", dpi=150)

# -------------------------------------------------------------
# PART 2: SIR model (3 ordinary differential equations)
#   dS/dt = -beta*S*I/N
#   dI/dt =  beta*S*I/N - gamma*I
#   dR/dt =  gamma*I
# -------------------------------------------------------------
print("\n=== PART 2: Epidemic spread (SIR model) ===")
N = ask("Total population (e.g. 1000): ", int, min_value=2)
I0 = ask("Number of people infected at the start (e.g. 1): ", int, min_value=1, max_value=N - 1)
infectious_days = ask("How many days does a sick person stay infectious? (e.g. 10): ", float, min_value=1)
R0_num = ask("On average, how many people does one sick person infect? (e.g. 3): ", float, min_value=0.1)

gamma = 1 / infectious_days
beta = R0_num * gamma
R_init = 0
S0 = N - I0 - R_init


def sir(y, t, beta, gamma, N):
    S, I, R = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]


def run(b, T):
    days = np.linspace(0, T, int(T * 10) + 1)
    return days, odeint(sir, [S0, I0, R_init], days, args=(b, gamma, N)).T


# Choose the simulation length automatically: extend until infected < 1 person
# (or until a 5-year cap), starting from 60 days.
T = 60.0
while True:
    days, (S, I, R) = run(beta, T)
    if (I[-1] < 1 and I.argmax() < len(I) - 1) or T >= 1825:
        break
    T *= 1.5
# trim the long flat tail after the epidemic ends
end_idx = np.where(I >= 1)[0]
end_idx = end_idx[-1] if len(end_idx) else len(I) - 1
cut = min(len(days) - 1, int(end_idx * 1.15) + 10)
days, S, I, R = days[:cut + 1], S[:cut + 1], I[:cut + 1], R[:cut + 1]
T = days[-1]

print(f"\nWorked out inside the model: gamma = 1/{infectious_days:g} = {gamma:.4f}, "
      f"beta = R0 x gamma = {beta:.4f}")
print(f"Simulated {T:.0f} days automatically.")
if R0_num <= 1:
    print("R0 <= 1, so the epidemic will die out without a large peak.")

plt.figure(figsize=(8, 5))
plt.plot(days, S, label="Susceptible (S)", color="blue")
plt.plot(days, I, label="Infected (I)", color="red")
plt.plot(days, R, label="Recovered (R)", color="green")
plt.title("SIR Model: Epidemic Spread")
plt.xlabel("Time (days)")
plt.ylabel("Number of people")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("2_sir_curves.png", dpi=150)

# -------------------------------------------------------------
# PART 3: Peak of the epidemic using derivatives 
#   dI/dt = 0  ->  I*(beta*S/N - gamma) = 0  ->  S = N*gamma/beta
# -------------------------------------------------------------
S_at_peak = N * gamma / beta
peak_idx = np.argmax(I)
print("\n--- Peak of epidemic (dI/dt = 0) ---")
print(f"Theory: dI/dt = 0 when S = N*gamma/beta = {S_at_peak:.1f}")
if S0 <= S_at_peak:
    print("S already starts at or below this value, so infections never rise.")
print(f"Numerical peak: day {days[peak_idx]:.1f}, infected = {I[peak_idx]:.1f}")
print(f"S at peak (numerical) = {S[peak_idx]:.1f}")

dI_dt = np.gradient(I, days)
plt.figure(figsize=(8, 4))
plt.plot(days, dI_dt, color="purple", label="dI/dt")
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(days[peak_idx], linestyle="--", color="red", label=f"Peak at day {days[peak_idx]:.0f}")
plt.title("Rate of change of infections (dI/dt = 0 at the peak)")
plt.xlabel("Time (days)")
plt.ylabel("dI/dt")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("3_peak_derivative.png", dpi=150)

# -------------------------------------------------------------
# PART 4: Total infected using integration (Unit IV)
#   area under I(t)           = total infection-days
#   integral of new infections = N - S(T) - R(0)
# -------------------------------------------------------------
trap = getattr(np, "trapezoid", None) or np.trapz
infection_days = trap(I, days)
total_infected_integral = trap(beta * S * I / N, days)
print("\n--- Total infected (integral of I) ---")
print(f"Area under I(t) (total infection-days) = {infection_days:.1f}")
print(f"Integral of new infections = {total_infected_integral:.1f}")
print(f"Check N - S(end) - R(0) = {N - S[-1] - R_init:.1f}")

plt.figure(figsize=(8, 4))
plt.plot(days, I, color="red")
plt.fill_between(days, I, alpha=0.3, color="red")
plt.title("Area under I(t) = total infection-days")
plt.xlabel("Time (days)")
plt.ylabel("Infected")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("4_integral_area.png", dpi=150)

# -------------------------------------------------------------
# PART 5: Flatten the curve (no input needed)
#   Contact reduced by 0%, 25%, 50%, 70%: new beta = beta*(1 - cut/100)
# -------------------------------------------------------------
print("\n=== PART 5: Flatten the curve (automatic) ===")
plt.figure(figsize=(8, 5))
for c in (0, 25, 50, 70):
    b = beta * (1 - c / 100)
    _, (_, I_b, _) = run(b, T)
    plt.plot(days, I_b[:len(days)], label=f"{c}% less contact  (R0 = {b / gamma:.1f})")
    print(f"  {c:>2}% less contact: R0 = {b / gamma:.2f}, peak infected = {I_b.max():.0f}")
plt.title("Effect of reducing contact (flattening the curve)")
plt.xlabel("Time (days)")
plt.ylabel("Infected")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("5_flatten_curve.png", dpi=150)

print("\nSaved 5 graphs as PNG files. Showing plots...")
plt.show()
