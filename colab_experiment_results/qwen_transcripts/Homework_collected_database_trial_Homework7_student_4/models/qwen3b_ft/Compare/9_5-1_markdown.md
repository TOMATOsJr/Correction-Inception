![Source Image](9_5-1_source.png)
 After \( t = 0 \), current source = 0.

\[
R \cdot (C \cdot s \cdot v_c) + L \cdot s \cdot ((L \cdot s \cdot v_c) + v_c) = 0
\]

\[
v_c(L \cdot C \cdot s^2 + R \cdot C \cdot s + 1) = 0
\]

\[
s^2 + \frac{R}{L} s + \frac{1}{LC} = 0 \quad \Rightarrow \quad s^2 + 4000s + 4 \cdot 10^6 = 0
\]

\[
s_{1,2} = -2000 \quad \Rightarrow \quad \textcolor{red}{(s + 2000)(s + 2000)} = 0
\]

\[
\therefore v_c(t) = A_1 e^{-2000t} + A_2 e^{-2000t}
\]

Before \( t = 0 \), current source = 2mA:

\[
i_c(0^-) = i_c(0^+) = 0A
\]

\[
v_c(0^-) = v_c(0^+) = (30 \cdot 10^{-3} A) \times (10 \Omega) = 3V
\]

\[
i_c(0^+) = \frac{\textcolor{red}{d}v_c(0^+)}{\textcolor{red}{dt}} \quad \Rightarrow \quad 0 = -2000A + A_2 \quad \Rightarrow \quad A_2 = 2000A
\]

\[
3 = A_1 + A_2 \quad \Rightarrow \quad 3 = A_1
\]

\[
A_2 = 2000 \, (\text{A}) = 6000 \, \text{mA}
\]

\[
v_c(t) = (3 + 6000) e^{-2000t} \, \text{V}
\]

