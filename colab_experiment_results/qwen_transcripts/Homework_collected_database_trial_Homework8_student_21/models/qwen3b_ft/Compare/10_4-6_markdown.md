![Source Image](10_4-6_source.png)
 

\[
I(\omega) = 120 \Omega = 120 \text{ mA}
\]

\[
Z_R = R = 40 \Omega
\]

\[
Z_L = j\omega L = j(4000)(15 \times 10^{-3} \text{ H}) = 60 \Omega
\]

\[
Z_C = \frac{1}{j\omega C} = \frac{1}{j(4000)(20 \times 10^{-6} \text{ F})} = -12.5 \Omega
\]

\[
\therefore I = I_1 + I_2
\]

\[
\therefore 120 \text{ mA} - I_2 = I_1
\]

\[
\therefore (60 + jL_2 + 40J_2 + 12.5j)I_1 = 0
\]

\[
\therefore (60 + jL_2 + 40J_2 + 12.5j)(120 \text{ mA} = I_2) = 0
\]

\[
\therefore (40 + 47.5j)I_2 = -1.5
\]

\[
\therefore I_2 = \frac{-1.5}{40 + 47.5j} = \frac{-1.5}{\sqrt{40^2 + 47.5^2}} = \frac{-1.5}{\sqrt{156.25}} = \frac{-1.5}{\textcolor{red}{6.17}} = \textcolor{red}{-0.2415} \text{ A}
\]

\[
\therefore V(\omega) = 40I_2 = 40(-\frac{57}{3095} - \frac{48}{3095}j) = \frac{\textcolor{red}{0.2415}}{\textcolor{red}{6.17}} = \textcolor{red}{0.9666} \text{ V}
\]

\[
\therefore V(t) = \textcolor{red}{0.9666} \cos(4000t - 140^\circ) \text{ V}
\]

