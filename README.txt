POPULATION GROWTH AND EPIDEMIC SPREAD USING THE SIR MODEL
Course: 25CS201 - Linear Algebra and Calculus for Computer Science Engineering
Micro-Project 13 (uses CO3 derivatives, CO4 integrals, CO5 differential equations)


HOW TO RUN
----------
1. Install Python 3 from python.org (tick "Add Python to PATH" while installing).

2. Unzip this folder. Open a terminal (Command Prompt on Windows) inside it.
   Tip: open the folder, click the address bar, type cmd and press Enter.

3. Install the libraries (only needed once):

       pip install -r requirements.txt

4. Run the program:

       python sir_model.py

5. Type the answers when asked and press Enter after each one.
   Five graphs are saved as PNG files in the same folder and also shown on screen.
   (Close each graph window to see the next one.)

If "python" is not recognised, try "py sir_model.py" (Windows) or "python3 sir_model.py" (Mac/Linux).


WHAT THE PROGRAM ASKS (SAMPLE ANSWERS YOU CAN TRY)
--------------------------------------------------
Part 1 - Population growth
  Starting population ........................................ 10
  Maximum population the area can support .................... 1000
  Growth per day when the population is small, in % .......... 50

Part 2 - Epidemic (SIR)
  Total population ........................................... 1000
  Number of people infected at the start ..................... 1
  Days a sick person stays infectious ........................ 10
  People one sick person infects on average .................. 3

You never have to type beta or gamma. The program works them out itself.
The number of days to simulate is also chosen automatically.
Part 5 (flatten the curve) needs no input.


OUTPUT FILES
------------
1_logistic_growth.png   Logistic population growth
2_sir_curves.png        S, I and R curves over time
3_peak_derivative.png   dI/dt crossing zero at the epidemic peak
4_integral_area.png     Area under the infected curve
5_flatten_curve.png     Effect of less contact between people


THE SIR MODEL
-------------
The population is split into three groups:
  S (Susceptible) - healthy people who can still catch the disease
  I (Infected)    - people who have it and can spread it
  R (Recovered)   - people who got better and are now immune

People move from S to I to R over time.

Equations:
  dS/dt = -beta * S * I / N
  dI/dt =  beta * S * I / N - gamma * I
  dR/dt =  gamma * I

  N = total population (S + I + R = N always)
  beta = infection rate, gamma = recovery rate
  R0 = beta / gamma. If R0 > 1 the epidemic spreads, if R0 < 1 it dies out.

How beta and gamma are found from the answers you type:
  gamma = 1 / (days a sick person stays infectious)
  beta  = R0 x gamma     (R0 = people one sick person infects on average)
  Example: 10 infectious days and R0 = 3 give gamma = 0.1 and beta = 0.3.


THE MATHS USED
--------------
Part 1: Logistic growth (differential equation, Unit V)
  dP/dt = r * P * (1 - P/K)
  P(t)  = K / (1 + ((K - P0)/P0) * e^(-r t))
  P = population, r = growth rate (percentage / 100), K = maximum population.
  The graph runs until the population reaches 99% of K.

Part 2: SIR system of three ordinary differential equations (Unit V)
  Solved numerically with SciPy odeint.

Part 3: Peak of the epidemic (derivatives, Unit III)
  The peak is where the slope of I(t) is zero:
    dI/dt = 0
    beta * S * I / N - gamma * I = 0
    I * (beta * S / N - gamma) = 0
    S = N * gamma / beta        (since I is not 0)
  Before the peak dI/dt > 0 (infections rising), after it dI/dt < 0 (falling).

Part 4: Total infected (integrals, Unit IV)
  Total infection-days = integral of I(t) dt from 0 to T   (area under the red curve)
  People newly infected = integral of beta*S*I/N dt = S(0) - S(T)
  Both are computed with the trapezoidal rule.

Part 5: Flatten the curve
  Less contact lowers beta, so R0 = beta / gamma falls and the peak is smaller and later.
  The program tries 0%, 25%, 50% and 70% less contact:
    new beta = beta * (1 - reduction / 100)


LIBRARIES USED
--------------
NumPy, SciPy (odeint), Matplotlib, SymPy
