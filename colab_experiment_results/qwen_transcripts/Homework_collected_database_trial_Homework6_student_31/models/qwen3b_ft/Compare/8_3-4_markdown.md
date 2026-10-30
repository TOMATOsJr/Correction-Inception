![Source Image](8_3-4_source.png)
 Before switch closes,

\[
12V \textcolor{red}{(5)} \quad 6\Omega \quad 3\Omega
\]

\[
i(0)
\]

\[
\therefore i(0) = \frac{12}{9} = \frac{4}{3} A
\]

Find \( V_{oc} \):

\[
2i_a - 12 = 6i_a
\]

\[
-4i_a = 12 \implies i_a = -3 A
\]

\[
V_{oc} = 2 \cdot i_a = 2 \cdot (-3) = -6 V
\]

Find \( I_{sc} \):

\[
I_{sc} = \frac{2i_a}{3} = \frac{-6}{3} = -2 A
\]

Find \( R_x \):

\[
R_x = \frac{V_{oc}}{I_{sc}} = \frac{-6}{-2} = 3 \Omega
\]

\[
T = \frac{T}{R_x} = \frac{6}{3} = 2 s
\]

\[
i(t) = i_x e^{-(t/2)} + (i(0) - i_x e^{-t/2}) e^{-0.5 t}
\]

\[
\therefore i(t) = -2 + \frac{10}{3} e^{-0.5 t} \quad \text{for } t > 0
\]

