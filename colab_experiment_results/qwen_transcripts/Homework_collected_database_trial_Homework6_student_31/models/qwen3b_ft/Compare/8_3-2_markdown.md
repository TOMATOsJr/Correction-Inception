![Source Image](8_3-2_source.png)
Before \( t = 0 \):
\[
i(0) = \frac{12}{6} = 2 \text{ A}
\]

Norton Equivalent Circuit:
\[
I_{sc} = \frac{12}{6 + 6} = 1 \text{ A}
\]
\[
R_x = 6 || 6 + 6 = 6 || 12 = \frac{1}{\frac{1}{6} + \frac{1}{12}} = 4 \Omega
\]
\[
t = \frac{L}{R_x} = \frac{8}{4} = 2 \text{ s}
\]
\[
i(t) = I_{sc} + (i(0) - I_{sc}) e^{-\frac{x}{R_x t}}
\]
\[
\Rightarrow i(t) = 1 + e^{-0.5 t} \quad \text{A for } t > 0
\]

