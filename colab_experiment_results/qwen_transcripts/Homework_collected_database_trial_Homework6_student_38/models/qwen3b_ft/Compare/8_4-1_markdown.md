![Source Image](8_4-1_source.png)
 

Capacitor -> open circuit, \( V_C(0) = 10 \, \text{V} \)

\( 0 < t < 1.5 \, \text{s}: \quad V_{AC} = \textcolor{red}{\text{Voltage divider}} \rightarrow \frac{8}{16} \cdot 10 = 5 \, \text{V} \)

\( R_L = \text{parallel resistors} = \frac{1}{8} + \frac{1}{8} = \frac{1}{4}, \quad R_L = 4 \, \Omega, \quad S = \frac{\textcolor{red}{\text{mV}}}{\frac{1}{T}} \cdot 0.05 \, \text{F} \, \text{c.c.} \)

\(\therefore V_L = -15 + 4i_L + v(t) = 0 \quad i(t) = C \frac{dV(t)}{dt} \rightarrow \frac{dV(t)}{dt} + 5v(t) = 25, \quad \tau = \frac{1}{5}, \quad k = 25 \)

\( v(t) = 5 + Ae^{-\tau t} \rightarrow 10 = 5 + Ae^{-\tau t} \quad A = 5 \quad v(t) = 5 + 5e^{-\tau t} \quad v(1.5) = 5, \quad 0.028 \, \text{V} \)

\( t = t - 1.5 \quad t > 1.5 \, \text{s}: \quad \textcolor{red}{\text{mV}} = \frac{8}{16} \cdot 0.05, \quad \textcolor{red}{\text{kV}} = -10 + \frac{1}{4} \frac{dV(t)}{dt} + v(t) = 0 \rightarrow \frac{dV(t)}{dt} + 2.5v(t) = 25 \)

\(\tau = \frac{1}{5}, \quad k = 25 \quad v(t') = 10 + Ae^{-\tau t'} \quad 0.028 = 10 + A, \quad A = -4.997 \quad \textcolor{red}{> 2} \)

\[
v(t) = 
\begin{cases}
v(t') = 5 + 5e^{-\tau t'} \, \text{V}, & 0 < t < 1.5 \, \text{s} \\
v(t') = 10 - 4.9972e^{-\frac{t}{\tau}(t - 1.5)} \, \text{V}, & t > 1.5 \, \text{s}
\end{cases}
\]

