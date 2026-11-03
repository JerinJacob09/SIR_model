import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os


days = np.linspace(0, 365, 3651)
run_number = 1

while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== Population growth ===")
    P0 = float(input("Starting population: "))
    K = float(input("Carrying capacity: "))
    growth_rate = float(input("Growth rate per day (%): ")) / 100

    print("\n=== SIR epidemic ===")
    N = int(input("Total population: "))
    I0 = int(input("Initially infected: "))
    infectious_days = float(input("Infectious period in days: "))
    R0 = float(input("Average people infected by one person (R0): "))

    if (not np.all(np.isfinite([P0, K, growth_rate, N, I0, infectious_days, R0]))
            or P0 <= 0 or K <= P0 or growth_rate <= 0 or N < 2
            or not 0 < I0 < N or infectious_days <= 0 or R0 <= 0):
        input("Invalid inputs. Please try again by pressing Enter.\n")
        continue

    def population_growth(P, t):
        return growth_rate * P * (1 - P / K)

    population = odeint(population_growth, P0, days).flatten()

    gamma = 1 / infectious_days
    beta = R0 * gamma
    S0 = N - I0

    def sir(y, t):
        S, I, R = y
        infections = beta * S * I / N
        recoveries = gamma * I
        return [-infections, infections - recoveries, recoveries]

    S, I, R = odeint(sir, [S0, I0, 0], days).T
    infection_days = np.trapezoid(I, days)
    new_infections = np.trapezoid(beta * S * I / N, days)

    print(f"\ngamma = {gamma:.4f}, beta = {beta:.4f}")
    print(f"Infection-days during simulation = {infection_days:.1f}")
    print(f"New infections during simulation = {new_infections:.1f}")

    population_file = f"1_population_{run_number}.png"
    sir_file = f"2_sir_{run_number}.png"
    infected_file = f"3_infected_{run_number}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(days, population, color="green")
    plt.axhline(K, color="gray", linestyle="--", label="Carrying capacity")
    plt.title("Population Growth")
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.legend()
    plt.savefig(population_file, dpi=150)
    print(f"Saved {population_file}. Close the graph to continue.")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(days, S, label="Susceptible")
    plt.plot(days, I, label="Infected")
    plt.plot(days, R, label="Recovered")
    plt.title("SIR Model")
    plt.xlabel("Time (days)")
    plt.ylabel("People")
    plt.legend()
    plt.savefig(sir_file, dpi=150)
    print(f"Saved {sir_file}. Close the graph to continue.")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(days, I, color="red")
    plt.fill_between(days, I, color="red", alpha=0.3)
    plt.title("Infected People (area = infection-days)")
    plt.xlabel("Time (days)")
    plt.ylabel("Infected people")
    plt.savefig(infected_file, dpi=150)
    print(f"Saved {infected_file}. Close the graph to continue.")
    plt.show()
    plt.close()

    if input("Run another simulation? (y/n): ").strip().lower() not in ("y", "yes"):
        break
    run_number += 1