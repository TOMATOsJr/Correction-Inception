![Source Image](9_3-3_source.png)
 

\[
\textcolor{red}{R_1' = 1 \Omega \quad \text{resistor}} \\
\textcolor{red}{R_2' = 2 \Omega \quad \text{resistor}} \\
\textcolor{red}{C = 10^{-6} \mu F \quad \text{capacitor}} \\
\textcolor{red}{L = 1 \text{ mH} \quad \text{inductor}} \\
\]

\[
V = V_C
\]

\[
i = C \frac{dV}{dt} \quad \text{or} \quad V = L \frac{di}{dt}
\]

\[
\frac{V_1}{R_1} = i_L + \frac{dV}{dt} \quad \Rightarrow \quad V_1 = V \quad \Rightarrow \quad C \frac{dV}{dt} = i_L. \quad \textcolor{red}{(1)}
\]

\[
V - L \frac{di}{dt} - R_1 i_L = 0. \quad \textcolor{red}{(2)}
\]

\[
V - L \left(\frac{V_1}{R_1} - C \frac{dV}{dt}\right) - R_2 \left(\frac{V_1}{R_1} - C \frac{dV}{dt}\right) = 0
\]

\[
V = \frac{1}{R_1} V_1 + \frac{L}{R_1} \frac{dV}{dt} + LC \frac{d^2V}{dt^2} - R_2 \frac{V_1}{R_1} + \frac{R_2}{R_1} V + R_2 C \frac{dV}{dt} = 0
\]

\[
V = \left(\frac{L}{R_1} + \frac{R_2}{R_1}\right) \frac{dV}{dt} + LC \frac{d^2V}{dt^2} + \frac{R_2}{R_1} V = \frac{R_2}{R_1} V_1
\]

\[
\frac{d^2V}{dt^2} + \left(\frac{1}{R_1 C} + \frac{R_2}{L}\right) \frac{dV}{dt} + \left(\frac{R_2}{R_1 C} + \frac{1}{L C}\right) V = \left(\frac{R_2}{R_1 C}\right) V_1
\]

\[
(S^2 + 102000 S + 3 \times 10^8) V = 2 \times 10^8 V_1
\]

\[
2 \alpha = 102000 \quad \Rightarrow \quad \alpha = 51000
\]

\[
w^2 = 3 \times 10^8 \quad \text{overdamped}
\]

\[
\sqrt{\lambda^2 - w_0^2} = \sqrt{51000^2 - 3 \times 10^8} = \sqrt{2301 \times 10^9} = 1000 \sqrt{2301}
\]

\[
\textcolor{red}{\text{Characteristic eqn.: } S^2 + 102000 S + 3 \times 10^8 = 0}
\]

\[
S_1 = -51000 + 1000 \sqrt{2301} = -3031
\]

\[
S_2 = -51000 - 1000 \sqrt{2301} = -98969
\]

