![Source Image](9_2-1_source.png)
 

\[
\textcolor{red}{KCL: } i_L = \frac{V}{R_2} + C \cdot \frac{dV}{dt}
\]

\[
\textcolor{red}{KVL: } -V_S + R_1 i_L + L \cdot \frac{dI}{dt} + V = 0
\]

\[
V_S = R_1 \cdot \frac{V}{R_2} + R_1 \cdot C \cdot \frac{dV}{dt} + V + L \left( \frac{dV}{dt} \cdot \frac{1}{R_2} + \left( \frac{dI}{dt} \right)^2 \right)
\]

\[
V_S = (L \cdot C) \frac{d^2V}{dt^2} + \left( \frac{1}{R_2} + R_1 \cdot C \right) \frac{dV}{dt} + \left( \frac{R_1}{R_2} + 1 \right)V
\]

\[
V_S = (1 \cdot 10^{-8}) \frac{d^2V}{dt^2} + (3 \cdot 10^{-5}) \frac{dV}{dt} + (1.02)V
\]

