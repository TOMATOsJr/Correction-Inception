![Source Image](9_2-1_source.png)
 

\[
KCL \quad \text{in} \quad \textcolor{red}{1} \quad \text{in} \quad \textcolor{red}{1}
\]

\[
L - \frac{V}{R_2} - C \frac{dv}{dt} = 0
\]

\[
\frac{di}{dt} + \frac{1}{R_1} \frac{dv}{dt} - C \frac{dv}{dt} = 0
\]

\[
KVL \quad \text{on left loop}
\]

\[
-V_3 + R_1 i_1 + L \frac{di}{dt} + R_2 i_2 = 0
\]

\[
V_3 = R_1 \left( \frac{V}{R_2} + C \frac{dv}{dt} \right) + L \frac{di}{dt} \left( \frac{V}{R_2} + C \frac{dv}{dt} \right) + V
\]

\[
V_3 = \left( \frac{R_1}{R_2} + 1 \right) V + \left( R_1 C + \frac{L}{R_2} \right) \frac{dv}{dt} + L \frac{d^2 v}{dt^2}
\]

\[
V_3 = \frac{51}{50} V + \left( 2(10E-6) + \frac{0.001}{100} \right) \frac{dv}{dt} + (0.001)(10E-6) \frac{d^2 v}{dt^2}
\]

