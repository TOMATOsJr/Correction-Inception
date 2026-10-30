![Source Image](9_4-1_source.png)
 

Determine \( V_C(t) \)

\[
\textcolor{red}{\frac{L}{i}} = \frac{\textcolor{red}{i}}{i} + \frac{V_C(t)}{R} = 6V, \quad \frac{dV_C}{dt}(0) = -3000V/s
\]

\[
\textcolor{red}{\text{So } i} = \frac{\textcolor{red}{i}}{i} \quad \text{and} \quad V_C(t) = L \textcolor{red}{i} + C \textcolor{red}{S}V_C = 1 \textcolor{red}{H}, \quad V_C(0) = 0
\]

\[
R = 80 \Omega, \quad C = 25 \mu F
\]

\[
V_C(t) = V_C - L \textcolor{red}{i} \textcolor{red}{i} = V - V_C + \frac{V}{R} + C \textcolor{red}{S}V_C = 0 - \textcolor{red}{i} \cdot \textcolor{red}{i}
\]

\[
\frac{V - V_C}{\textcolor{red}{i}} = \textcolor{red}{i} \cdot \textcolor{red}{i}
\]

\[
V - V_C + \frac{51V}{R} + C S^2 V = 0
\]

\[
C S^2 V + \frac{6}{R} 51V - V = V_C
\]

\[
\left( S^2 + \frac{1}{R} CS + \frac{1}{LC} \right) V = V_C / R
\]

\[
(S^2 + 500S + 40000) V = 0
\]

\[
\alpha = 250; \quad \omega_0^2 = 40,000
\]

\[
\sqrt{\alpha^2 - \omega_0^2} = \sqrt{250^2 - 40,000} = \sqrt{22500} = 150
\]

\[
S = -250 + 150 = -100
\]

\[
S_2 = -250 - 150 = -400
\]

\[
V_{2,0} = A_1 e^{-100t} + A_2 e^{-400t}
\]

\[
V_C(t) = 0
\]

\[
V_C(t) = A_1 e^{-100t} + A_2 e^{-400t}
\]

Solve for \( A_1 \) and \( A_2 \).

\[
\textcolor{red}{C} = A_1 + A_2 \quad \Rightarrow \quad \textcolor{red}{6} - A_1 = A_2
\]

\[
\frac{dV_C}{dt} = -100A_1 e^{-100t} - 400A_2 e^{-400t}
\]

\[
-3000 = -100A_1 - 400A_2
\]

\[
-3000 = -100A_1 - 400(\textcolor{red}{6} - A_1) \quad \Rightarrow \quad -3000 = 300A_1 - 2400
\]

\[
-600 = 300A_1
\]

\[
\textcolor{red}{6} - (\textcolor{red}{-2}) = A_1 = 8
\]

\[
A_1 = -2
\]

\[
V_C(t) = -2e^{-100t} + 8e^{-400t}, \quad t > 0
\]

