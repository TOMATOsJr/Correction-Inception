![Source Image](9_2-1_source.png)
 

\[
V = L \frac{di}{dt}, \quad i = C \frac{dv}{dt}
\]

KCL at top node

\[
i_L = \frac{V}{R_2} + C \frac{dv}{dt}
\]

\[
-V_S + R_1 i_L + L \frac{di}{dt} + V = 0
\]

\[
-V_S + R_1 \left( \frac{V}{R_2} + C \frac{dv}{dt} \right) + L \frac{d}{dt} \left( \frac{V}{R_2} + C \frac{dv}{dt} \right) + V = 0
\]

\[
-V_S + \frac{R_1 i_L}{R_2} + R_1 C \frac{dv}{dt} + \frac{L}{R_2} \frac{dv}{dt} + LC \frac{d^2 v}{dt^2} + V = 0
\]

\[
LC \frac{d^2 v}{dt^2} + \left( R_1 C + \frac{L}{R_2} \right) \frac{dv}{dt} + \left( \frac{R_1}{R_2} + 1 \right) V = V_S
\]

\[
\frac{d^2 v}{dt^2} + \left( \frac{R_1}{L} + \frac{1}{CR_2} \right) \frac{dv}{dt} + \left( \frac{R_1}{R_2} + \frac{1}{LC} \right) V = \frac{1}{LC} V_S
\]

\[
\frac{d^2 v}{dt^2} + 3000 \frac{dv}{dt} + 1.02 \times 10^8 V = 1 \times 10^8 V_S
\]

