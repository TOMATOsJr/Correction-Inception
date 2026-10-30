![Source Image](8_3-5_source.png)
For \( t < 0 \): 
\[
\textcolor{red}{\text{Study state, open circuit, so } i' = 0}
\]
\[
\textcolor{red}{\text{Voltage drop across } R = 0, \text{ so } V_o(0) = 5 \text{ V}}
\]

For \( t > 0 \): 
\[
\textcolor{red}{\text{KCL @ top node out: } \frac{5}{20 \Omega} + \frac{5 - V_{out}}{20 \Omega} = 0 \quad V_{out} = 10 \text{ V}}
\]
\[
\textcolor{red}{\text{Assuming short circuit will @ top node out: } \frac{5}{20 \Omega} + \frac{5 - V_{out}}{20 \Omega} = 0 \quad V_{out} = 10 \text{ V}}
\]
\[
i_s = \frac{V_{out}}{R} = 0.005 \text{ A}
\]
\[
R_t = \frac{V_{out}}{i_s} = 20 \text{ k}\Omega
\]
\[
\therefore 10 \text{ V} @ \frac{5}{20 \text{ k}\Omega} = 4 \mu \text{F}
\]
\[
\text{KVL: } -10 + 200000i + V_o(t) = 0, \quad i = C \frac{dV_o(t)}{dt}
\]
\[
\frac{dV_o}{dt} + \frac{25}{4} V_o(t) = 125
\]
\[
T = \frac{2}{25} \text{ s} = 125 \text{ ms}
\]
\[
V_o(t) = 10 - A e^{-\frac{25}{4}t} \text{ V}
\]
\[
V_o(0) = S = 10 - A, \quad A = 5
\]
\[
V_o(t) = 10 - 5 e^{-\frac{25}{4}t} \text{ V}
\]

