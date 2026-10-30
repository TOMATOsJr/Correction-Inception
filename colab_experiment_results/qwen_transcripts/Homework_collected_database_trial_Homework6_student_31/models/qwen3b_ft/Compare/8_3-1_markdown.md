![Source Image](8_3-1_source.png)
 Voltage division formula

Thevenin equivalent circuit:

\[
V_{\text{oc}} = 12 \cdot \frac{6}{6 + 6} = 6 \, \text{V}
\]

\[
R_{\text{T}} = \frac{1}{\frac{1}{6} + \frac{1}{6}} = 3 \, \Omega
\]

\[
I = R_{\text{T}} \cdot C = 3 \cdot 0.25 = 0.75 \, \text{A}
\]

\[
v(t) = V_{\text{oc}} + (V(0) - V_{\text{oc}}) e^{-t/RC}
\]

\[
\Rightarrow v(t) = 6 - 2 e^{1.33 t} \, \text{V} \quad \text{for } t > 0
\]

