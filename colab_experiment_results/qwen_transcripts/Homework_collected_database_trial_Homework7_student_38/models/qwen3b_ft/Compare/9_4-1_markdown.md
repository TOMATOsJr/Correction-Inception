![Source Image](9_4-1_source.png)
 

KCL at top node in: \( i_L - \frac{V}{R_0} - (\frac{dV}{dt}) = 0 \rightarrow i_L = \frac{V}{R_0} + (\frac{dV}{dt}) \)

KVL left mesh: \( -V_S + L \frac{di}{dt} + R_0 i_L - (\frac{dV}{dt}) = 0 \)

Operator method: \( i_L = \frac{V}{R_0} + (LSV) \cdot (-V_S + L \frac{di}{dt} + R_0 i_L - (\frac{dV}{dt})) = 0 \)

\( \therefore -V_S + LS \frac{V}{R_0} + LLS^2 V + V = 0 \quad \textcolor{red}{LLS^2 V + \frac{1}{R_0} LSV + V = V_S} \quad S^2 V + \frac{1}{R_0} LSV + V = V_S \quad S^2 V + 500 LSV + 40000 V = 40000 V_S \quad \therefore S^2 + 500 S + 40000 = 0 \quad \alpha = 280 \, \omega^2 \, 4000 \)

Roots: \( s_1 = -100, \, s_2 = -400 \quad V_S = 0 \) for \( t \geq 0 \). So complete response = neutral response.

\( V(t) = A_1 e^{-100t} + A_2 e^{-400t} \quad \frac{dV}{dt} = -100 A_1 e^{-100t} - 400 A_2 e^{-400t} \)

\[
\begin{cases}
A_1 + A_2 = 6 \\
-100 A_1 - 400 A_2 = -3000 \\
\end{cases}
\]

\[
\Rightarrow A_1 = -2, \quad A_2 = 8
\]

\[
V(t) = -2 e^{-100t} + 8 e^{-400t}, \quad V, \, t \geq 0
\]

