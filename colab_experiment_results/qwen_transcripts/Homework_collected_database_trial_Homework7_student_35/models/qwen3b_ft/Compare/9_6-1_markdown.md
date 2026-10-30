![Source Image](9_6-1_source.png)
 

(1) 
\[
\frac{V}{250} + i_1 + 5 \cdot 10^{-5} \frac{dV}{dt} = 0
\]
\[
V_c = 0.8 \frac{V}{\textcolor{red}{\text{m}}}
\]

\[
\textcolor{red}{\frac{d^2 V}{dt^2}} + 30 \frac{dV}{dt} + 7.5 \cdot 10^5 V = 0
\]

\[
\textcolor{red}{s^2} - 300 s + 250000 = 0
\]
\[
s = -400 \pm 300j
\]

\[
V_c(t) = e^{-400t} \left( 3e^{-300t} + 4e^{-75000t} \right) V
\]

\[
V_1(t) = e^{-400t} \left( 4e^{-300t} + 4e^{-75000t} \right)
\]

\[
V(0) = 250 \left( \frac{-400}{500} \right) + 6 = 3V = V_c(0)
\]

From (1):
\[
\frac{dV_c(0)}{dt} = 3 = i_1
\]
\[
\frac{dV_c(0)}{dt} - V = -400 \cdot 0.8 + 300 \cdot 4
\]
\[
\frac{dV_c(0)}{dt} = 4
\]

