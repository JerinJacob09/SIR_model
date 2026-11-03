import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp

def ask(prompt, cast=float, min_value=None, max_value=None):
    while True:
        try:
            value = cast(input(prompt).strip())
            if not math.isfinite(value):
                raise ValueError
        except ValueError:
            print("  Please enter a valid number.")
            continue
        if min_value is not None and value < min_value:
            print(f"  Value must be at least {min_value}.")
        elif max_value is not None and value > max_value:
            print(f"  Value must be at most {max_value}.")
        else:
            return value

def save_and_show(filename):
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}. Close the graph window to continue.")
    plt.show()

print("=== Population growth ===")
P0 = ask("Starting population : ", float, min_value=1)
K_val = ask("Maximum population the area can support : ", float, min_value=P0 + 1)
while P0 >= 0.99 * K_val:
    print("  K must be greater than P0/0.99 so the 99% target is still ahead.")
    K_val = ask("Maximum population the area can support : ", float, min_value=P0 + 1)
growth_pct = ask("Growth per day when the population is small, in % : ", float, min_value=0.01)
r_val = growth_pct / 100

t_s = sp.symbols("t")
P = sp.Function("P")
r, K = sp.symbols("r K", positive=True)

ode = sp.Eq(P(t_s).diff(t_s), r * P(t_s) * (1 - P(t_s) / K))
particular_sol = sp.dsolve(ode, ics={P(0): P0}).subs({r: r_val, K: K_val})

print("\nLogistic growth analytical solution P(t):")
sp.pprint(particular_sol)

ratio = 99 * (K_val - P0) / P0
t_end = float(np.log(ratio) / r_val) if ratio > 1 else 10.0

t = np.linspace(0, t_end, 300)
A = (K_val - P0) / P0
P_t = K_val / (1 + A * np.exp(-r_val * t))

plt.figure(figsize=(7, 4))
plt.plot(t, P_t, color="green")
plt.axhline(K_val, linestyle="--", color="gray", label="Maximum population K")
plt.title("Logistic Population Growth")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_and_show("1_logistic_growth.png")

print("\n=== Epidemic spread (SIR model) ===")
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

def run(b, T=365):
    days = np.linspace(0, T, T * 10 + 1)
    S, I, R = odeint(sir, [S0, I0, R_init], days, args=(b, gamma, N)).T
    return days, S, I, R

days, S, I, R = run(beta)

end_day = days[np.where(I > 0.01 * I.max())[0][-1]]

print(f"\nWorked out inside the model: gamma = 1/{infectious_days:g} = {gamma:.4f}, "
      f"beta = R0 x gamma = {beta:.4f}")
print("Simulated 365 days.")
if I[-1] >= 1:
    print("NOTE: the epidemic is still going after 365 days, so the peak and totals below are incomplete.")
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
plt.xlim(0, end_day)
save_and_show("2_sir_curves.png")
print("\n--- Peak of epidemic (dI/dt = 0) ---")
S_at_peak = N * gamma / beta
peak_idx = np.argmax(I)
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
plt.xlim(0, end_day)
save_and_show("3_peak_derivative.png")
print("\n--- Infections during the simulated period ---")
infection_days = (R[-1] - R_init) / gamma
new_infections = S0 - S[-1]

print(f"Area under I(t) = integral of I dt = (R(end) - R(0))/gamma = {infection_days:.1f} infection-days")
print(f"New infections = integral of beta*S*I/N dt = S(0) - S(end) = {new_infections:.1f}")
print(f"Ever infected since the start, including I0 = {I0 + new_infections:.1f}")
print(f"Check using S + I + R = N: I(end) + R(end) - I0 - R(0) = {I[-1] + R[-1] - I0 - R_init:.1f}")

plt.figure(figsize=(8, 4))
plt.plot(days, I, color="red")
plt.fill_between(days, I, alpha=0.3, color="red")
plt.title("Area under I(t) = total infection-days")
plt.xlabel("Time (days)")
plt.ylabel("Infected")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.xlim(0, end_day)
save_and_show("4_integral_area.png")
print("\n=== Flatten the curve ===")
plt.figure(figsize=(8, 5))
for c in (0, 25, 50, 70):
    b = beta * (1 - c / 100)
    _, _, I_b, _ = run(b)
    plt.plot(days, I_b, label=f"{c}% less contact  (R0 = {b / gamma:.1f})")
    if np.argmax(I_b) == 0:
        print(f"  {c:>2}% less contact: R0 = {b / gamma:.2f}, no epidemic (infections only fall)")
    else:
        print(f"  {c:>2}% less contact: R0 = {b / gamma:.2f}, peak infected = {I_b.max():.0f} on day {days[np.argmax(I_b)]:.1f}")
plt.title("Effect of reducing contact (flattening the curve)")
plt.xlabel("Time (days)")
plt.ylabel("Infected")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_and_show("5_flatten_curve.png")
print("\nDone. All 5 graphs were saved as PNG files.")
