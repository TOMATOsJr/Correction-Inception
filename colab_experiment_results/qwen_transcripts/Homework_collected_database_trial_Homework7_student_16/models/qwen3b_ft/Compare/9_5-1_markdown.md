![Source Image](9_5-1_source.png)
 

After:

\[
\textcolor{red}{\begin{cases}
100 \Omega \\
\frac{1}{250 \mu F} \\
-100 i + 250 \frac{di}{dt} + V_c = 0 \\
i = 100 e^{-\frac{8}{R} t} \\
-\textcolor{red}{0.01} \frac{dV_c}{dt} + 250 \textcolor{red}{0.0005} \frac{di}{dt} + V_c = 0 \\
\frac{d^2 V_c}{dt^2} + 400 \frac{di}{dt} + 400 \textcolor{red}{0.0005} V_c = 0 \\
s^2 + 400 s + 400 \textcolor{red}{0.0005} = 0 \\
s_1 = -2000 \\
s_2 = -2000 \\
V_c(t) = A e^{-2000t} + B t e^{-2000t}
\end{cases}}
\]

Before:

\[
\textcolor{red}{\begin{cases}
V_c = 30 \text{mA} \cdot 100 \\
= 3V \\
V_c(0) = |3| = A \\
\frac{dV_c}{dt}(0) = 0 = -2000A + B \\
\Rightarrow -6000 + B = 0 \\
\therefore B = 6000 \\
V_c(t) = 3 e^{-2000t} + 6000 e^{-2000t} \textcolor{red}{V}
\end{cases}}
\]

