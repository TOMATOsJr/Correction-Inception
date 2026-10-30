![Source Image](8_3-2_source.png)
 Steady-state inductor is short circuit, current only passes through bottom resistor:

\[
v = Ri \quad i = \frac{12}{R} = 2 \, A \quad i(0) = 2 \, A
\]

Thévenin's Equivalent after \( t = 0 \):

- Voltage divider: \( V_{oc} = \frac{6}{18} \cdot 12 = 4 \, V \)
- \( R_L \): Two series, then per \( RL, \frac{1}{R_1} + \frac{1}{R_2} = \frac{1}{R_L} \Rightarrow R_L = 4 \Omega \)

\[
\textcolor{red}{\text{3 SHM}} = L
\]

KVL: 
\[
-4 + 4(i(t) + v_L(t)) = 0 \quad \Rightarrow \quad v_L(t) = L \frac{d(i(t))}{dt}
\]
\[
\frac{d(i(t))}{dt} + \frac{1}{2} i(t) = \frac{1}{2}
\]

\[
T = 2 \, \text{ms} = \frac{1}{2} \quad \quad i(t) = 1 + 4 e^{-\frac{t}{4}} \quad 2 = 1 + A \quad A = 1 \quad (i_L(t) = 1 + e^{-\frac{t}{4}} \quad A)
\]

