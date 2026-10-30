![Source Image](8_3-10_source.png)
Closed for a long time, inductor is short-circuited

\[
\textcolor{red}{L = 70 \, \Omega}
\]

\[
\textcolor{red}{R_1 = 14 \, \Omega}
\]
\[
\textcolor{red}{R_2 = 28 \, \Omega}
\]
\[
\textcolor{red}{R_3 = 3 \, \Omega}
\]

Left side of circuit: \( V_{oc} = \textcolor{red}{15 \, V} \)

\[
R_{eq} = \frac{\textcolor{red}{R_1}}{\textcolor{red}{R_1} + \textcolor{red}{R_2}} = \frac{1}{3} \rightarrow 3 \, \Omega
\]

\[
\therefore 15 \, V \textcolor{red}{\odot} 3 \, \Omega \textcolor{red}{\odot} 3 \, \Omega \rightarrow \frac{\textcolor{red}{15}}{3} \, A
\]

\[
\textcolor{red}{KCL @ top node in:} \quad \frac{\textcolor{red}{15} - v}{3} - \frac{v}{4} + i_9 = 0
\]

\[
\therefore v = 18 \, V
\]

Current through branch: \(\frac{\textcolor{red}{15}}{3} = 5 \, A\)

\[
\textcolor{red}{\text{Current divider: } i_9 = \frac{R_2}{R_1 + R_2} i = \frac{3}{12} \cdot 8 = 2 \, A = i_1 (0)}
\]

\[
t > 0: \quad \frac{\textcolor{red}{12} i(t)}{3} \, \text{A} \quad \textcolor{red}{KVL:} \quad 12 i(t) + v(t) = 0 \quad \Rightarrow \quad \frac{d(v(t))}{dt} + 24 i(t) = 0
\]

\[
\therefore \frac{dv}{dt} = \frac{1}{24} \, \text{A} = 0
\]

\[
\therefore i_1(t) = A e^{-\frac{1}{24} t} \quad \textcolor{red}{i_1(0) = 2 \, A = A e^{0}}
\]

\[
\therefore A = 2
\]

\[
v(t) = 9 \cdot i_1(t)
\]

\[
\therefore v(t) = 18 e^{-\frac{1}{24} t} \, V
\]

