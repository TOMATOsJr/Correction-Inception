![Source Image](9_5-1_source.png)
 After \( t \geq 0 \)

\[
i = C \frac{dV}{dt} = 30 \text{ mA}
\]

\[
R_i + L \frac{di}{dt} + V_e = 0
\]

\[
L \cdot S \cdot i + R \cdot (CSV) + V = 0
\]

\[
LC^2 \cdot V + RC \cdot SV + V = 0
\]

\[
(S^2 + \frac{R}{L} + \frac{1}{T})V = 0
\]

\[
S^2 + 4000S^2 + 4 \times 10^6 = 0
\]

\[
\alpha = 2000, \quad \omega_0^2 = 4 \times 10^6
\]

\[
\sqrt{4 \times 10^6 \cdot 4 \times 10^6} = \sqrt{0} = 0 \quad \text{(Critically damped)}
\]

\[
S_1 \cdot S_2 = -2000
\]

No forced response since \( \textcolor{red}{\text{t}(t) = 0} \).

\[
V_n = V(t) = (A_1 + A_2)e^{-2000t} \textcolor{red}{V}
\]

\[
\frac{3}{4} = A_2
\]

\[
C \frac{d^2V}{dt^2} = 0 \quad \Rightarrow \quad A_1 e^{-2000t} = 2000(A_1 + A_2)e^{-2000t}
\]

\[
0 = A_1 - 6000
\]

\[
6000 = A_1
\]

\[
V(t) = (6000(1 + 3)e^{-2000t}) \textcolor{red}{V}, \quad t \geq 0
\]

