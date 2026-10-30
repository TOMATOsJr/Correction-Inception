![Source Image](9_5-1_source.png)
 

\[
R = 1 \Omega + L \frac{di}{dt} + V_C = 0
\]

\[
i_c = 10 \text{ W/m}^2 = \frac{\textcolor{red}{i}}{dt}
\]

\[
\therefore 0.001 \frac{\textcolor{red}{i}}{dt} + 2.5 \cdot 10^{-7} \frac{di}{dt} + V_C = 0
\]

\[
\therefore s^2 + 4000s + 4 \cdot 10^6 = 0
\]

\[
s = -20000
\]

\textcolor{red}{general solution}: \( i(t) = A_1 e^{-2000t} + A_2 t e^{-2000t} \)

\[
i(0) = 3
\]

\[
\therefore A_2 = 6000
\]

\[
\therefore i(t) = (6000t + 3) e^{-2000t}
\]

