![Source Image](9_5-1_source.png)
 Also assuming \(\frac{dv_c}{dt} = 0, t > 0\):

\[
\frac{d}{dt}(100\textcolor{red}{\text{sign}}(i_L)) = \frac{d}{dt}(v_c)
\]

KVL: \(v_c + 100i_L + L\frac{di_L}{dt} = 0\)

LL \(\frac{d}{dt}(v_c) + 100L\frac{dv_c}{dt} + v_c = 0\)

\(\therefore s^2 v_c + 4000sv_c + 4 \cdot 10^6 v_c = 0\).

\(\alpha = 2000\), \(\omega_0^2 = 4 \cdot 10^6\), \(s_1 = -2000\), \(s_2 = -2000\).

\(v_c(t) = e^{-2000t}(A_1t + A_2)\)

\(v_c(0) = 3 = A_2\)

\(\therefore v_c(t) = e^{-2000t}(6000t + 3)\) for \(t > 0\).

\(\frac{dv_c}{dt} = -2000(A_1t + A_2)e^{-2000t} + A_1e^{-2000t}\).

\(\therefore 0 = -6000 + A_1\), \(A_1 = 6000\).

