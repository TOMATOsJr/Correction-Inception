![Source Image](8_7-3_source.png)
 Capacitor -> Open circuit, \( V_c(0) = \textcolor{red}{\text{Voltage across } 5k\Omega} \). Overtail \( t = 0 \), so \( V_c(0) = 0 \).

\( t > 0 \): \( V_{oc} = R_i i' = 5 \times 1000 \cdot 12 \cos 2t \cdot 10^{-3} = 10 \cos 2t \, V \). \( R_i = 15 \, k\Omega \).

\( 10 \cos 2t \, V \) @ \( \frac{15 \, k\Omega}{\frac{1}{2} \, \text{F}} = 10 \, \mu\text{F} \).

KVL: \( -10 \cos 2t + 15000 i + V = 0 \). \( i = \left( \frac{\textcolor{red}{\frac{V}{2t}}}{\frac{1}{2} \, \text{F}} \right) \).

\[
\frac{dV}{dt} + 2V = 20 \cos 2t \quad \text{at} \quad V(0) = A \sin(2t) + B \cos(2t)
\]

\(\tau = \frac{1}{2} \).

\( 2A \cos 2t - 2B \sin 2t + 2A \sin(2t) + 2B \cos(2t) = 20 \cos(2t) \).

\( 2A + 2B = 20 \). \( 2A - 2B = 0 \). \( A = 5 \), \( B = 5 \).

\( V_c(t) = 4 e^{-2t} + 5 \sin(2t) + 5 \cos(2t) \, V \), \( 0 < t < \frac{1}{2} \).

\( V_c(t) = 5 \sin(2t) + 5 \cos(2t) - 5 e^{-2t} \, V \), \( t > \frac{1}{2} \).

