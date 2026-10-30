![Source Image](8_7-3_source.png)
 Method of Undetermined Coefficients

\[
V_n = Ae^{-zt}
\]

\[
V_1 = B \sin(zt) + C \cos(zt)
\]

Plug into filter:

\[
20 \omega_0 \textcolor{red}{(2t)} - 2 \sin(zt) + 2 \left[ B \sin(zt) + C \cos(zt) \right] = \frac{2}{3} \cdot \frac{1}{c} \cos(zt)
\]

\[
(-2C + 2B) \sin(zt) + (2B + 2C) \cos(zt) = \frac{2}{3} \cdot \frac{1}{c} \cos(zt)
\]

\[
-2C + 2B = 0 \implies C = B
\]

\[
2B + 2C = \frac{2}{3} \cdot \frac{1}{c} \cdot 10^{-3} \implies 2B + 2C = \frac{2}{10} \cdot 10^{-3}
\]

\[
2B + 2C = 2 \cdot 10^{-4}
\]

\[
B + C = 10^{-4}
\]

\[
2B = 10^{-4} \implies B = 5000 = 5000
\]

\[
V_C(0) = 0 = A e^{0} + 5000 \sin(0) + 5000 \cos(0)
\]

\[
0 = A + 5000
\]

\[
A = -5000
\]

\(\therefore V_C(t) = -5000 e^{-zt} + 5000 \sin(zt) + 5000 \cos(zt)\)

