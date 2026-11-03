import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp


def ask(prompt, cast=float, min_value=None, max_value=None):
    # Keep asking until the user types a finite number within the bounds.
    while True:
        text = input(prompt).strip()
        try:
            value = cast(text)
            if cast is float and not math.isfinite(value):
                raise ValueError
        except (ValueError, OverflowError):
            expected = "whole number" if cast is int else "finite number"
            print(f"  Please enter a valid {expected}.")
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
while P0 >= 0.99 * K_val:
    print("  K must be greater than P0/0.99 so the 99% target is still ahead.")
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


MAX_SIM_DAYS = 1825.0
TAIL_CASE_TOLERANCE = 0.05
TAIL_INFECTION_DAYS_TOLERANCE = 0.05


def run(b):
    # Each beta gets its own horizon; the remaining-case bound uses the current R_t.
    T = 60.0
    while True:
        days = np.linspace(0, T, int(T * 10) + 1)
        S, I, R = odeint(sir, [S0, I0, R_init], days, args=(b, gamma, N)).T
        peak_passed = np.argmax(I) < len(I) - 1
        effective_R = b * S[-1] / (gamma * N)
        remaining_cases = (
            I[-1] * effective_R / (1 - effective_R)
            if effective_R < 1 else np.inf
        )
        remaining_infection_days = (I[-1] + remaining_cases) / gamma
        finished = (
            I[-1] < 1 and peak_passed
            and remaining_cases <= TAIL_CASE_TOLERANCE
            and remaining_infection_days <= TAIL_INFECTION_DAYS_TOLERANCE
        )
        hit_cap = T >= MAX_SIM_DAYS
        if finished or hit_cap:
            return days, (S, I, R), hit_cap and not finished
        T = min(T * 1.5, MAX_SIM_DAYS)


days, (S, I, R), baseline_incomplete = run(beta)
T = days[-1]

print(f"\nWorked out inside the model: gamma = 1/{infectious_days:g} = {gamma:.4f}, "
      f"beta = R0 x gamma = {beta:.4f}")
print(f"Simulated {T:.0f} days automatically.")
if baseline_incomplete:
    print("WARNING: the 5-year cap was reached before the epidemic tail was negligible; final totals may be incomplete.")
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
# PART 3: Peak of the epidemic using derivatives (Unit III)
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

dI_dt = beta * S * I / N - gamma * I
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
#   integral of new infections after t=0 = S(0) - S(T)
# -------------------------------------------------------------
trap = getattr(np, "trapezoid", None) or np.trapz
infection_days = trap(I, days)
new_infections = trap(beta * S * I / N, days)
print("\n--- Infections during the simulated period ---")
print(f"Area under I(t) (total infection-days) = {infection_days:.1f}")
print(f"New infections after the start (integral of beta*S*I/N) = {new_infections:.1f}")
print(f"Check S(start) - S(end) = {S0 - S[-1]:.1f}")
print(f"Ever infected since the start, including I0 = {I0 + new_infections:.1f}")
print(f"Check I0 + new infections = N - S(end) - R(0) = {N - S[-1] - R_init:.1f}")

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
    if c == 0:
        days_b, I_b, incomplete = days, I, baseline_incomplete
    else:
        days_b, (_, I_b, _), incomplete = run(b)
    plt.plot(days_b, I_b, label=f"{c}% less contact  (R0 = {b / gamma:.1f})")
    peak_b = int(np.argmax(I_b))
    peak_label = "peak observed by cap" if incomplete else "peak infected"
    print(f"  {c:>2}% less contact: R0 = {b / gamma:.2f}, {peak_label} = {I_b[peak_b]:.0f} on day {days_b[peak_b]:.1f}")
    if incomplete:
        print("     WARNING: this scenario reached the 5-year cap before the epidemic tail was negligible.")
plt.title("Effect of reducing contact (flattening the curve)")
plt.xlabel("Time (days)")
plt.ylabel("Infected")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("5_flatten_curve.png", dpi=150)

print("\nSaved 5 graphs as PNG files. Showing plots...")
plt.show()
